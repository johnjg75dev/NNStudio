import Icon from '../Icon';
import { Badge, Button, EmptyState } from '../ui';
import { useSession, useSessionStore } from '../../state/SessionContext';
import { fmtNum } from '../../lib/format';

function stamp(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
}

/** History: an undo trail of every change made to the setup. */
export default function HistoryPanel() {
  const store = useSessionStore();
  const history = useSession((s) => s.history);
  const stepsRun = useSession((s) => s.stepsRun);

  if (!history.length) {
    return (
      <EmptyState
        icon="history"
        title="No changes recorded yet"
        message="Every edit you make to the task, layers or hyperparameters is logged here so you can step back to any earlier setup."
      />
    );
  }

  return (
    <div className="history">
      <div className="row" style={{ gap: 8, marginBottom: 10 }}>
        <Badge tone="accent">{history.length} recorded change{history.length === 1 ? '' : 's'}</Badge>
        <Badge mono>{stepsRun} steps run</Badge>
        <Button
          size="xs"
          variant="ghost"
          icon="trash"
          style={{ marginLeft: 'auto' }}
          onClick={() => store.set({ history: [] })}
        >
          Clear
        </Button>
      </div>

      <ol className="history__list">
        {history.map((h, i) => (
          <li key={h.id} className="history__item">
            <span className="history__dot" aria-hidden="true" />
            <div className="history__body">
              <div className="row wrap" style={{ gap: 6 }}>
                <strong className="tiny">{h.action}</strong>
                <Badge mono>{stamp(h.timestamp)}</Badge>
                {i === 0 && <Badge tone="pos">latest</Badge>}
              </div>
              <div className="row wrap" style={{ gap: 8, marginTop: 4 }}>
                <span className="mono xs muted">{h.config.archKey}</span>
                <span className="mono xs muted">{h.config.funcKey}</span>
                <span className="mono xs muted">[{(h.config.layers || []).map((l) => l.neurons ?? l.type).join('·')}]</span>
                <span className="mono xs muted">lr {fmtNum(h.config.lr)}</span>
                <span className="mono xs muted">{h.config.optimizer}</span>
              </div>
            </div>
            <Button
              size="xs"
              variant="ghost"
              iconOnly
              title="Restore this setup"
              onClick={() => store.restoreHistory(h.id)}
            >
              <Icon name="history" />
            </Button>
          </li>
        ))}
      </ol>

      <p className="tiny muted">
        <Icon name="info" size={12} /> Restoring rebuilds the network with the recorded setup —
        weights are re-initialised.
      </p>
    </div>
  );
}
