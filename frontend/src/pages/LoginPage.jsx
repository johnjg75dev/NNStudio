import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import api from '../api/client';
import { Button, Field, TextInput } from '../components/ui';
import Icon from '../components/Icon';

export default function LoginPage() {
  const [params] = useSearchParams();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    if (!username || !password) {
      setError('Enter both a username and a password.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.submitAuthForm('/login', { username, password });
      const next = params.get('next');
      // Full navigation: every provider re-mounts with an authenticated session.
      window.location.assign(next && next.startsWith('/') && !next.startsWith('/login') ? next : '/train');
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <form className="auth__form" onSubmit={onSubmit} noValidate>
      <Field label="Username" htmlFor="login-username">
        <TextInput
          id="login-username"
          value={username}
          onChange={setUsername}
          autoComplete="username"
          autoFocus
          placeholder="e.g. ada"
        />
      </Field>

      <Field label="Password" htmlFor="login-password">
        <TextInput
          id="login-password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete="current-password"
          placeholder="••••••••"
        />
      </Field>

      {error && (
        <div className="info info--neg row" style={{ gap: 8 }}>
          <Icon name="alert" size={14} />
          <span>{error}</span>
        </div>
      )}

      <Button type="submit" variant="primary" size="lg" block loading={busy} icon="logout">
        Sign in
      </Button>

      <p className="tiny muted center" style={{ marginTop: 4 }}>
        New here? <Link to="/signup">Create an account</Link> — it seeds your presets, architectures
        and starter datasets.
      </p>
    </form>
  );
}
