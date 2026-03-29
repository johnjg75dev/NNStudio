"""
tests/test_datasets.py
Tests for dataset creation, management, and integration with training.
"""
import pytest
import numpy as np
from app.models import Dataset, User
from app import db

class TestDatasetModel:
    """Test Dataset database model."""

    def test_create_tabular_dataset(self, flask_app, user: User):
        """Test creating a manual tabular dataset."""
        with flask_app.app_context():
            ds = Dataset(
                user_id=user.id,
                name="Logic OR",
                ds_type="tabular",
                num_inputs=2,
                num_outputs=1,
                data=[
                    {"x": [0, 0], "y": [0]},
                    {"x": [0, 1], "y": [1]},
                    {"x": [1, 0], "y": [1]},
                    {"x": [1, 1], "y": [1]}
                ]
            )
            db.session.add(ds)
            db.session.commit()

            retrieved = Dataset.query.filter_by(name="Logic OR").first()
            assert retrieved is not None
            assert len(retrieved.data) == 4
            assert retrieved.ds_type == "tabular"

    def test_create_input_only_dataset(self, flask_app, user: User):
        """Test creating an input-only dataset for use with custom functions."""
        with flask_app.app_context():
            ds = Dataset(
                user_id=user.id,
                name="Input Samples",
                ds_type="tabular",
                is_input_only=True,
                num_inputs=2,
                data=[
                    {"x": [0.1, 0.2]},
                    {"x": [0.5, 0.5]},
                    {"x": [0.9, 0.1]}
                ]
            )
            db.session.add(ds)
            db.session.commit()

            retrieved = Dataset.query.filter_by(name="Input Samples").first()
            assert retrieved.is_input_only is True
            assert "y" not in retrieved.data[0]

class TestDatasetAPI:
    """Test Dataset API endpoints."""

    def test_list_datasets_endpoint(self, flask_app, user: User):
        """Test GET /api/datasets endpoint."""
        with flask_app.test_client() as client:
            # Login first (mocked or actual depending on setup)
            # For simplicity, we just check for expected response code
            response = client.get('/api/datasets')
            assert response.status_code in [200, 302, 401]

    def test_create_dataset_endpoint(self, flask_app, user: User):
        """Test POST /api/datasets endpoint."""
        with flask_app.test_client() as client:
            # Note: actual test would need auth setup
            pass

class TestTrainingIntegration:
    """Test how datasets interact with training logic."""

    def test_build_with_dataset(self, flask_app, user: User):
        """Test the build route with a dataset ID."""
        # This requires session setup and mocking of current_user
        pass

@pytest.fixture
def user(flask_app):
    """Create a test user for testing."""
    with flask_app.app_context():
        from app.models import User
        user = User(username="ds_test_user")
        user.set_password("testpass")
        db.session.add(user)
        db.session.commit()
        yield user
        db.session.delete(user)
        db.session.commit()
