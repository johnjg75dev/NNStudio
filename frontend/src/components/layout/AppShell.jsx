import { useEffect, useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import Icon from '../Icon';
import { Badge, Button, StatusPill } from '../ui';
import ShortcutsDialog from './ShortcutsDialog';
import { useCatalog } from '../../state/CatalogContext';
import { useSession, useSessionActions, useSessionStore } from '../../state/SessionContext';
import { useTheme } from '../../state/ThemeContext';
import { useHotkeys } from '../../lib/hooks';
import { fmtCompact, fmtLoss, fmtPct } from '../../lib/format';

export const NAV_ITEMS = [
  { to: '/train', label: 'Studio', icon: 'network', hint: 'Build and train a network' },
  { to: '/datasets', label: 'Datasets', icon: 'database', hint: 'Create and edit datasets' },
  { to: '/playground', label: 'Playground', icon: 'beaker', hint: 'Probe the model by hand' },
  { to: '/models', label: 'Models', icon: 'folder', hint: 'Save, load and export models' },
  { to: '/functions', label: 'Functions', icon: 'code', hint: 'Write custom training tasks' },
  { to: '/learn', label: 'Learn', icon: 'book', hint: 'Architecture & concept library' },
];

const PAGE_META = {
  '/train': { title: 'Training Studio', subtitle: 'Design the network, then watch it learn' },
  '/datasets': { title: 'Datasets', subtitle: 'Your training data — tabular, image or generated' },
  '/playground': { title: 'Playground', subtitle: 'Run forward passes and sweep inputs by hand' },
  '/models': { title: 'Model Library', subtitle: 'Save, restore and export what you trained' },
  '/functions': { title: 'Custom Functions', subtitle: 'Teach the studio a new task in Python or JS' },
  '/learn': { title: 'Learn', subtitle: 'Architecture diagrams, activations and concept notes' },
};

export default function AppShell() {
  const navigate = useNavigate();
  const location = useLocation();
  const catalog = useCatalog();
  const store = useSessionStore();
  const actions = useSessionActions();
  const { theme, toggle } = useTheme();

  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);

  const status = useSession((s) => s.status);
  const statusMessage = useSession((s) => s.statusMessage);
  const metrics = useSession((s) => s.metrics);
  const running = useSession((s) => s.running);
  const dirty = useSession((s) => s.configDirty);

  // ── auth gate ──
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchMe();
        if (cancelled) return;
        if (!data?.authenticated) {
          navigate(`/login?next=${encodeURIComponent(location.pathname)}`, { replace: true });
          return;
        }
        setUser(data);
        setAuthChecked(true);
      } catch {
        if (!cancelled) navigate('/login', { replace: true });
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── boot the training session once the catalogue has arrived ──
  useEffect(() => {
    if (!authChecked || catalog.loading) return;
    store.boot(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked, catalog.loading]);

  useHotkeys({
    Space: () => actions.toggleTrain(),
    KeyR: () => actions.reset(),
    KeyB: () => actions.build(),
    Slash: (e) => {
      if (e.shiftKey) setShowShortcuts(true);
    },
    Question: () => setShowShortcuts(true),
    KeyT: () => toggle(),
    Digit1: () => navigate('/train'),
    Digit2: () => navigate('/datasets'),
    Digit3: () => navigate('/playground'),
    Digit4: () => navigate('/models'),
    Digit5: () => navigate('/functions'),
    Digit6: () => navigate('/learn'),
  });

  if (!authChecked) {
    return (
      <div className="boot">
        <div className="boot__card">
          <span className="spinner spinner--lg" />
          <div>
            <div className="strong">Starting NNStudio…</div>
            <div className="tiny muted">Loading your workspace and catalogues</div>
          </div>
        </div>
      </div>
    );
  }

  const meta = PAGE_META[location.pathname] || { title: 'NNStudio', subtitle: '' };
  const statusTone = running ? 'active' : status === 'error' ? 'error' : status === 'paused' ? 'warn' : '';

  return (
    <div className="shell">
      <nav className="rail" aria-label="Primary">
        <div className="rail__brand" title="NNStudio">
          <Icon name="logo" size={22} />
        </div>
        <div className="rail__nav">
          {NAV_ITEMS.map((item, i) => (
            <a
              key={item.to}
              href={item.to}
              className="rail__item"
              data-active={location.pathname === item.to ? 'true' : 'false'}
              data-label={`${item.label}  ·  ${i + 1}`}
              title={item.hint}
              aria-current={location.pathname === item.to ? 'page' : undefined}
              onClick={(e) => {
                e.preventDefault();
                navigate(item.to);
              }}
            >
              <Icon name={item.icon} size={20} />
            </a>
          ))}
        </div>

        <div className="rail__spacer" />

        <button
          className="rail__item"
          data-label="Keyboard shortcuts  ·  ?"
          onClick={() => setShowShortcuts(true)}
          title="Keyboard shortcuts"
        >
          <Icon name="keyboard" size={19} />
        </button>
        <button
          className="rail__item"
          data-label={theme === 'dark' ? 'Light theme  ·  T' : 'Dark theme  ·  T'}
          onClick={toggle}
          title="Toggle theme"
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={19} />
        </button>
        <a className="rail__item" href="/logout" data-label="Sign out" title={`Sign out (${user?.username})`}>
          <Icon name="logout" size={19} />
        </a>
      </nav>

      <div className="shell__main">
        <header className="topbar">
          <div className="topbar__title">
            <h1>{meta.title}</h1>
            <span>{meta.subtitle}</span>
          </div>

          <div className="topbar__spacer" />

          <div className="topbar__group">
            {dirty && (
              <Badge tone="warn" title="Your configuration changed since the last build">
                rebuild pending
              </Badge>
            )}
            <LiveMetrics metrics={metrics} running={running} />
            <StatusPill tone={statusTone}>{statusMessage || status}</StatusPill>
            <Button
              variant={running ? 'default' : 'success'}
              size="sm"
              icon={running ? 'pause' : 'play'}
              onClick={() => actions.toggleTrain()}
            >
              {running ? 'Pause' : 'Train'}
            </Button>
            <span className="topbar__user" title={user?.username}>
              <Icon name="user" size={13} />
              {user?.username}
            </span>
          </div>
        </header>

        <div className="shell__body">
          <Outlet />
        </div>
      </div>

      <ShortcutsDialog open={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </div>
  );
}

function LiveMetrics({ metrics, running }) {
  return (
    <div className="livestats" aria-live="off">
      <span className="livestats__item">
        <i>epoch</i>
        <b>{metrics.epoch ?? 0}</b>
      </span>
      <span className="livestats__item">
        <i>loss</i>
        <b>{fmtLoss(metrics.loss)}</b>
      </span>
      <span className="livestats__item">
        <i>acc</i>
        <b>{metrics.accuracy === null || metrics.accuracy === undefined ? '—' : fmtPct(metrics.accuracy)}</b>
      </span>
      <span className="livestats__item">
        <i>params</i>
        <b>{fmtCompact(metrics.params)}</b>
      </span>
      {running && <span className="livestats__live" title="Training in progress" />}
    </div>
  );
}

async function fetchMe() {
  const res = await fetch('/api/me', { credentials: 'same-origin' });
  const json = await res.json();
  return json?.data ?? null;
}
