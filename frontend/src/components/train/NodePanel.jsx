import { useMemo } from 'react';
import Icon from '../Icon';
import ActivationChart from '../canvas/ActivationChart';
import { Badge, Button, EmptyState } from '../ui';
import { ACTIVATION_BY_KEY } from '../../lib/activations';
import { nodeInfo, traceInfluence } from '../../lib/influence';
import { useSession, useSessionStore } from '../../state/SessionContext';
import { fmtNum, fmtPct } from '../../lib/format';
import { layerLabel } from '../../lib/layers';

const MAX_BARS = 20;

/** Node inspector: everything about the neuron clicked on the canvas. */
export default function NodePanel({ onExploreLatent }) {
  const store = useSessionStore();
  const node = useSession((s) => s.selectedNode);
  const snapshot = useSession((s) => s.snapshot);
  const focusMode = useSession((s) => s.focusMode);

  const info = useMemo(() => nodeInfo(snapshot, node), [snapshot, node]);
  const influences = useMemo(
    () => (focusMode && node ? traceInfluence(snapshot, node) : []),
    [focusMode, node, snapshot],
  );

  if (!node || !info) {
    return (
      <EmptyState
        icon="cursor"
        title="No neuron selected"
        message="Click any neuron on the network canvas to inspect its activation, bias and weights. Shift-click — or press Trace — to follow its influences back to the inputs."
        action={
          (snapshot?.topology?.length || 0) > 2 ? (
            <Button size="sm" icon="cursor" onClick={() => store.selectNode({ layer: 1, idx: 0 })}>
              Inspect the first hidden neuron
            </Button>
          ) : null
        }
      />
    );
  }

  const act = info.activationName ? ACTIVATION_BY_KEY[info.activationName] : null;
  const topology = snapshot?.topology || [];

  return (
    <div className="node">
      <header className="node__head">
        <div className="row wrap" style={{ gap: 6 }}>
          <Badge tone={info.type === 'input' ? 'cyan' : info.type === 'output' ? 'violet' : 'accent'}>
            {info.type}
          </Badge>
          <Badge mono>
            col {node.layer} · n{node.idx}
          </Badge>
          {info.layerType && <Badge>{layerLabel(info.layerType)}</Badge>}
        </div>
        <Button size="xs" variant="ghost" iconOnly title="Clear selection" onClick={() => store.clearSelection()}>
          <Icon name="close" />
        </Button>
      </header>

      <div className="kv">
        <div className="kv__row">
          <span className="kv__k">Activation</span>
          <span className="kv__v mono strong">{info.value === null ? '—' : fmtNum(info.value)}</span>
        </div>
        {info.bias !== null && !Number.isNaN(info.bias) && (
          <div className="kv__row">
            <span className="kv__k">Bias</span>
            <span className="kv__v mono">{fmtNum(info.bias)}</span>
          </div>
        )}
        {act && (
          <div className="kv__row">
            <span className="kv__k">Function</span>
            <span className="kv__v">
              {act.name} <span className="mono tiny muted">{act.formula}</span>
            </span>
          </div>
        )}
        {info.nIn !== null && (
          <div className="kv__row">
            <span className="kv__k">Incoming</span>
            <span className="kv__v mono">
              {info.nIn} weights
              {info.fanOut ? ` · feeds ${info.fanOut}` : ''}
            </span>
          </div>
        )}
        <div className="kv__row">
          <span className="kv__k">Network</span>
          <span className="kv__v mono">[{topology.join(' → ')}]</span>
        </div>
      </div>

      {act && (
        <div className="node__curve">
          <ActivationChart type={info.activationName} height={104} marker={info.value} />
          <p className="tiny muted">{act.use}</p>
        </div>
      )}

      <div className="row wrap" style={{ gap: 6, margin: '10px 0' }}>
        {info.type === 'hidden' && (
          <Button size="sm" variant="primary" icon="wave" onClick={() => onExploreLatent?.(node)}>
            Explore latent space
          </Button>
        )}
        {info.type !== 'input' && (
          <Button
            size="sm"
            variant={focusMode ? 'warn' : 'ghost'}
            icon="focus"
            onClick={() => store.toggleFocus()}
          >
            {focusMode ? 'Stop tracing' : 'Trace influences'}
          </Button>
        )}
      </div>

      {focusMode && (
        <section className="node__influences">
          <header className="sec-head">
            <h3>What drives this neuron</h3>
            <Badge mono>{influences.length} upstream</Badge>
          </header>
          {!influences.length ? (
            <p className="tiny muted">Nothing upstream to trace from here.</p>
          ) : (
            <ul className="influence-list">
              {influences.map((inf, i) => (
                <li key={`${inf.layer}-${inf.idx}`}>
                  <button
                    type="button"
                    className="influence"
                    title={`Column ${inf.layer}, neuron ${inf.idx} — ${inf.depth} hop(s) back`}
                    onClick={() => store.selectNode({ layer: inf.layer, idx: inf.idx })}
                  >
                    <span className="influence__rank mono">{i + 1}</span>
                    <span className="influence__id mono">
                      c{inf.layer}·n{inf.idx}
                    </span>
                    <span className="influence__bar">
                      <span className="influence__fill" style={{ width: `${Math.min(100, inf.influence * 100)}%` }} />
                    </span>
                    <span className="influence__val mono">{fmtPct(inf.influence)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <p className="tiny muted mt-8">
            Influence follows |weights| backwards from this neuron, scaled by how active each upstream
            neuron currently is. Click a row to jump to it — the canvas dims everything else.
          </p>
        </section>
      )}

      {info.weightsIn.length > 0 && (
        <section className="node__weights">
          <header className="sec-head">
            <h3>Incoming weights</h3>
            <Badge mono>{info.weightsIn.length}</Badge>
          </header>
          <div className="weight-bars">
            {info.weightsIn.slice(0, MAX_BARS).map((w, i) => {
              const v = Number(w) || 0;
              return (
                <div key={i} className="weight-bar">
                  <span className="mono xs muted">n{i}</span>
                  <span className="weight-bar__track">
                    <span
                      className={`weight-bar__fill ${v >= 0 ? 'pos' : 'neg'}`}
                      style={
                        v >= 0
                          ? { left: '50%', width: `${Math.min(50, Math.abs(v) * 20)}%` }
                          : { right: '50%', width: `${Math.min(50, Math.abs(v) * 20)}%` }
                      }
                    />
                  </span>
                  <span className={`mono xs ${v >= 0 ? 'pos' : 'neg'}`}>{fmtNum(v)}</span>
                </div>
              );
            })}
          </div>
          {info.weightsIn.length > MAX_BARS && (
            <p className="tiny muted">…and {info.weightsIn.length - MAX_BARS} more (see Weight matrices).</p>
          )}
        </section>
      )}

      {info.weightsOut.length > 0 && (
        <section className="node__weights">
          <header className="sec-head">
            <h3>Outgoing weights</h3>
            <Badge mono>{info.weightsOut.length}</Badge>
          </header>
          <div className="row wrap" style={{ gap: 4 }}>
            {info.weightsOut.map((w, i) => (
              <span key={i} className={`io__chip mono ${Number(w) >= 0 ? 'pos' : 'neg'}`}>
                n{i} {fmtNum(w)}
              </span>
            ))}
          </div>
        </section>
      )}

      {info.gradientOut.some((g) => Number.isFinite(g) && Math.abs(g) > 0) && (
        <p className="tiny muted mt-8">
          <Icon name="bolt" size={12} /> Gradient reaching this neuron:{' '}
          <span className="mono">{fmtNum(Math.hypot(...info.gradientOut.map((g) => Number(g) || 0)))}</span>
        </p>
      )}
    </div>
  );
}
