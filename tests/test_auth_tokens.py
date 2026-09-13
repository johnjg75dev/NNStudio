"""
tests/test_auth_tokens.py
Cookie-free authentication.

The studio is usually opened inside an iframe on someone else's origin (a live
preview), where browsers may refuse to *store* the Flask session cookie even
with `SameSite=None; Secure`.  Login therefore also hands out a signed token
that the SPA keeps in localStorage and replays as `X-Session-Token`.

These tests drive that path with a client that has no cookies at all, which is
exactly what the embedded browser ends up doing.
"""
import pytest

from app import db
from app.auth_tokens import make_token, read_token
from app.models import Dataset, Preset, User


@pytest.fixture
def user(flask_app):
    """A persisted user to sign in as."""
    with flask_app.app_context():
        u = User(username="token_user")
        u.set_password("testpass")
        db.session.add(u)
        db.session.commit()
        yield u


def login(client, username="token_user", password="testpass"):
    """POST /login the way the SPA does: form body, JSON answer."""
    return client.post(
        "/login",
        data={"username": username, "password": password},
        headers={"Accept": "application/json"},
    )


class TestTokenSigning:
    def test_roundtrip(self, flask_app):
        with flask_app.app_context():
            token = make_token(42)
            assert isinstance(token, str) and token
            assert read_token(token) == 42

    def test_rejects_garbage(self, flask_app):
        with flask_app.app_context():
            assert read_token("not-a-token") is None
            assert read_token("") is None
            assert read_token(None) is None

    def test_rejects_a_token_signed_with_another_secret(self, flask_app):
        from itsdangerous import URLSafeTimedSerializer
        from app.auth_tokens import SALT

        forged = URLSafeTimedSerializer("a-different-secret", salt=SALT).dumps(1)
        with flask_app.app_context():
            assert read_token(forged) is None


class TestLoginPayload:
    def test_login_returns_a_token(self, client, user):
        res = login(client)
        assert res.status_code == 200
        body = res.get_json()
        assert body["ok"] is True
        assert body["username"] == "token_user"
        assert body["token"]

    def test_bad_password_is_rejected(self, client, user):
        res = login(client, password="wrong")
        assert res.status_code == 401
        assert res.get_json()["ok"] is False

    def test_signup_returns_a_token_and_seeds_the_account(self, client):
        res = client.post(
            "/signup",
            data={"username": "brand_new", "password": "testpass"},
            headers={"Accept": "application/json"},
        )
        assert res.status_code == 200
        token = res.get_json()["token"]
        assert token

        with client.application.app_context():
            new_user = User.query.filter_by(username="brand_new").first()
            assert new_user is not None
            assert Preset.query.filter_by(user_id=new_user.id).count() > 0
            assert Dataset.query.filter_by(user_id=new_user.id).count() > 0


class TestCookieFreeRequests:
    """A brand-new client with an empty cookie jar, authenticating by header."""

    def test_token_authenticates_api_me(self, client, flask_app, user):
        token = login(client).get_json()["token"]

        cookieless = flask_app.test_client()
        res = cookieless.get("/api/me", headers={"X-Session-Token": token})
        assert res.status_code == 200
        data = res.get_json()["data"]
        assert data["authenticated"] is True
        assert data["username"] == "token_user"

    def test_token_unlocks_protected_library_routes(self, client, flask_app, user):
        token = login(client).get_json()["token"]

        cookieless = flask_app.test_client()
        for path in ("/api/datasets", "/api/models", "/api/functions/custom"):
            res = cookieless.get(path, headers={"X-Session-Token": token})
            assert res.status_code == 200, path

    def test_query_param_fallback(self, client, flask_app, user):
        token = login(client).get_json()["token"]

        cookieless = flask_app.test_client()
        res = cookieless.get(f"/api/me?session_token={token}")
        assert res.get_json()["data"]["authenticated"] is True

    def test_cookie_session_still_works(self, client, user):
        """Nothing about the classic cookie flow changes."""
        assert login(client).status_code == 200
        res = client.get("/api/me")
        assert res.get_json()["data"]["authenticated"] is True


class TestAnonymousApi:
    def test_protected_api_answers_401_json_not_a_redirect(self, client):
        """fetch() would follow a 302 and then fail to parse the login HTML."""
        res = client.get("/api/datasets", headers={"Accept": "application/json"})
        assert res.status_code == 401
        body = res.get_json()
        assert body["ok"] is False
        assert "Authentication" in body["error"]

    def test_public_api_stays_open(self, client):
        assert client.get("/api/modules/all").status_code == 200
        res = client.get("/api/me")
        assert res.status_code == 200
        assert res.get_json()["data"]["authenticated"] is False

    def test_bad_token_is_not_an_authenticated_session(self, client):
        res = client.get("/api/datasets", headers={"X-Session-Token": "garbage"})
        assert res.status_code == 401


class TestTrainingSessionKeying:
    """
    The in-memory training session used to be keyed by a random id stored in the
    session cookie.  Without a cookie that id changed on every request, so the
    network was rebuilt from scratch each time.  A signed-in user is now keyed by
    their account id.
    """

    def test_training_progress_survives_across_requests(self, client, flask_app, user):
        token = login(client).get_json()["token"]
        headers = {"X-Session-Token": token, "Accept": "application/json"}
        cookieless = flask_app.test_client()

        built = cookieless.post(
            "/api/session/build",
            json={
                "func_key": "xor",
                "arch_key": "mlp",
                "optimizer": "adam",
                "lr": 0.05,
                "loss": "bce",
                "layers": [{"type": "dense", "neurons": 4, "activation": "relu"}],
            },
            headers=headers,
        )
        assert built.status_code == 200
        assert built.get_json()["data"]["topology"] == [2, 4, 1]

        first = cookieless.post("/api/train/step", json={"steps": 5, "lr": 0.05}, headers=headers)
        second = cookieless.post("/api/train/step", json={"steps": 5, "lr": 0.05}, headers=headers)
        assert first.status_code == 200 and second.status_code == 200

        # Epochs accumulate instead of restarting at 5 on every call.
        assert second.get_json()["data"]["epoch"] == 10

        snapshot = cookieless.get("/api/session/snapshot", headers=headers)
        assert snapshot.get_json()["data"]["built"] is True
        assert snapshot.get_json()["data"]["epoch"] == 10

    def test_two_users_get_separate_sessions(self, flask_app, user):
        with flask_app.app_context():
            other = User(username="second_user")
            other.set_password("testpass")
            db.session.add(other)
            db.session.commit()
            other_token = make_token(other.id)

        from app.api.helpers import get_session_id

        with flask_app.test_request_context("/", headers={"X-Session-Token": make_token(user.id)}):
            from flask_login import login_user

            login_user(user)
            first_key = get_session_id()

        with flask_app.test_request_context("/", headers={"X-Session-Token": other_token}):
            from flask_login import login_user

            login_user(other)
            second_key = get_session_id()

        assert first_key != second_key
        assert first_key == f"user:{user.id}"
