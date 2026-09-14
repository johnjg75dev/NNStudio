import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import { Button, Field, TextInput } from '../components/ui';
import Icon from '../components/Icon';

export default function SignupPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [availability, setAvailability] = useState(null); // { available, message }
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);

  // Debounced username availability check.
  useEffect(() => {
    clearTimeout(timer.current);
    if (username.trim().length < 3) {
      setAvailability(null);
      return undefined;
    }
    timer.current = setTimeout(async () => {
      try {
        const res = await api.checkUsername(username.trim());
        setAvailability(res);
      } catch {
        setAvailability(null);
      }
    }, 420);
    return () => clearTimeout(timer.current);
  }, [username]);

  const strength = passwordStrength(password);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    if (username.trim().length < 3) return setError('Username must be at least 3 characters.');
    if (availability && !availability.available) return setError(availability.message);
    if (password.length < 6) return setError('Password must be at least 6 characters.');
    if (password !== confirm) return setError('The two passwords do not match.');

    setBusy(true);
    try {
      await api.submitAuthForm('/signup', { username: username.trim(), password });
      window.location.assign('/train');
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <form className="auth__form" onSubmit={onSubmit} noValidate>
      <Field label="Username" htmlFor="su-username">
        <TextInput
          id="su-username"
          value={username}
          onChange={setUsername}
          autoComplete="username"
          autoFocus
          placeholder="pick something memorable"
        />
      </Field>
      {availability && username.trim().length >= 3 && (
        <div
          className="row tiny"
          style={{ gap: 6, color: availability.available ? 'var(--pos)' : 'var(--neg)', marginTop: -6 }}
        >
          <Icon name={availability.available ? 'check' : 'error'} size={12} />
          {availability.message}
        </div>
      )}

      <Field label="Password" htmlFor="su-password" hint="At least 6 characters.">
        <TextInput
          id="su-password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete="new-password"
          placeholder="••••••••"
        />
      </Field>
      {password && (
        <div className="row" style={{ gap: 6, marginTop: -6 }}>
          <div className="progress grow">
            <div
              className="progress__bar"
              style={{
                width: `${strength.pct}%`,
                background: strength.color,
              }}
            />
          </div>
          <span className="tiny" style={{ color: strength.color }}>
            {strength.label}
          </span>
        </div>
      )}

      <Field label="Confirm password" htmlFor="su-confirm">
        <TextInput
          id="su-confirm"
          type="password"
          value={confirm}
          onChange={setConfirm}
          autoComplete="new-password"
          placeholder="••••••••"
        />
      </Field>

      {error && (
        <div className="info info--neg row" style={{ gap: 8 }}>
          <Icon name="alert" size={14} />
          <span>{error}</span>
        </div>
      )}

      <Button type="submit" variant="primary" size="lg" block loading={busy} icon="sparkles">
        Create account
      </Button>

      <p className="tiny muted center" style={{ marginTop: 4 }}>
        Already have one? <Link to="/login">Sign in instead</Link>.
      </p>
    </form>
  );
}

function passwordStrength(pw) {
  let score = 0;
  if (pw.length >= 6) score += 1;
  if (pw.length >= 10) score += 1;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score += 1;
  if (/\d/.test(pw)) score += 1;
  if (/[^A-Za-z0-9]/.test(pw)) score += 1;
  const table = [
    { label: 'too short', color: 'var(--neg)', pct: 12 },
    { label: 'weak', color: 'var(--neg)', pct: 28 },
    { label: 'fair', color: 'var(--warn)', pct: 50 },
    { label: 'good', color: 'var(--warn)', pct: 70 },
    { label: 'strong', color: 'var(--pos)', pct: 88 },
    { label: 'excellent', color: 'var(--pos)', pct: 100 },
  ];
  return table[Math.min(score, 5)];
}
