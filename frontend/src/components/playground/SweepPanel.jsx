import { useMemo } from 'react';
import Icon from '../Icon';
import Plot2DCanvas from '../canvas/Plot2DCanvas';
import { Badge, Button, Field, NumberInput } from '../ui';
import { useSession, useSessionActions } from '../../state/SessionContext';
import { fmtNum, truncateArray } from '../../lib/format';

const MAX_EDITABLE = 3;
const MAX_ROWS = 200;

/**
 * SweepPanel — evaluate the network across a grid of input values instead of a
 * single point. Two-input tasks also get the decision-boundary plot.
 */
export default function SweepPanel() {
  const actions = useSessionActions();
  const snapshot = useSession((s) => s.snapshot);
  const sweep = useSession((s) => s.sweep);
  const busy = useSession((s) => s.busy);
  const func = snapshot?.func || null;
  const ranges = sweep?.ranges || [];
  const results = sweep?.results || null;
  const inputs = func?.inputs ?? ranges.length;

  const points = useMemo(
    () =>
      ranges.reduce((acc, r) => {
        const span = Number(r.max) - Number(r.min);
        const step = Number(r.step) || 0.1;
        if (span <= 0 || step <= 0) return acc;
        return acc * (Math.floor(span / step) + 1);
      }, 1),
    [ranges],
  );

  if (!snapshot?.built) return null;

  const setRange = (i, patch) => {
    const next = ranges.map((r, idx) => (idx === i ? { ...r, ...patch } : r));
    actions.setSweepRanges(next);
  };

  const tooMany = points > 5000;

  return (
    <div className="sweep">
      <div className="row wrap" style={{ gap: 8, marginBottom: 10 }}>
        <Badge tone={tooMany ? 'warn' : 'accent'} mono>
          {points.toLocaleString()} grid points
        </Badge>
        <Badge mono>{inputs} inputs</Badge>
        <Button
          size="sm"
          variant="primary"
          icon="sweep"
          loading={busy}
          style={{ marginLeft: 'auto' }}
          onClick={() => actions.runSweep()}
        >
          Run sweep
        </Button>
        {results && (
          <Button size="sm" variant="ghost" icon="trash" onClick={() => actions.setSweepRanges(ranges)}>
            Reset ranges
          </Button>
        )}
      </div>

      <div className="sweep__ranges">
        {ranges.slice(0, MAX_EDITABLE).map((r, i) => (
          <div key={i} className="sweep__row">
            <span className="sweep__label tiny truncate" title={func?.input_labels?.[i] || `x${i}`}>
              {func?.input_labels?.[i] || `x${i}`}
            </span>
            <Field label="min" inline className="field--tight">
              <NumberInput className="input--xs mono" value={r.min} step={r.step} onChange={(v) => setRange(i, { min: v })} />
            </Field>
            <Field label="max" inline className="field--tight">
              <NumberInput className="input--xs mono" value={r.max} step={r.step} onChange={(v) => setRange(i, { max: v })} />
            </Field>
            <Field label="step" inline className="field--tight">
              <NumberInput className="input--xs mono" value={r.step} step={r.step} min={0.001} onChange={(v) => setRange(i, { step: Math.max(0.001, v) })} />
            </Field>
          </div>
        ))}
        {ranges.length > MAX_EDITABLE && (
          <p className="tiny muted">
            The remaining {ranges.length - MAX_EDITABLE} inputs stay fixed at their current values — a grid
            over more than three dimensions explodes combinatorially.
          </p>
        )}
      </div>

      {tooMany && (
        <div className="banner banner--warn">
          <Icon name="alert" size={14} />
          <span>
            {points.toLocaleString()} predictions is a lot — raise the step size to keep the sweep snappy.
          </span>
        </div>
      )}

      {inputs <= 2 && (
        <div className="sweep__plot">
          <header className="sec-head">
            <h3>Response surface</h3>
            <Badge tone="violet">live from the server</Badge>
          </header>
          <Plot2DCanvas />
        </div>
      )}

      {results && results.length > 0 && (
        <div className="sweep__results">
          <header className="sec-head">
            <h3>Grid results</h3>
            <Badge mono>{results.length} rows</Badge>
          </header>
          <div className="table-scroll table-scroll--short">
            <table className="table table--io">
              <thead>
                <tr>
                  <th>Inputs</th>
                  <th>Predicted</th>
                  <th className="center">Top</th>
                </tr>
              </thead>
              <tbody>
                {results.slice(0, MAX_ROWS).map((r, i) => (
                  <tr key={i}>
                    <td className="mono tiny truncate" title={(r.x || []).map(fmtNum).join(', ')}>
                      {truncateArray(r.x, 6).join(' · ')}
                    </td>
                    <td className="mono tiny truncate" title={(r.pred || []).map(fmtNum).join(', ')}>
                      {truncateArray(r.pred, 6).join(' · ')}
                    </td>
                    <td className="center mono tiny">
                      {(r.pred || []).length > 1
                        ? (r.pred || []).indexOf(Math.max(...r.pred))
                        : fmtNum((r.pred || [])[0])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {results.length > MAX_ROWS && (
            <p className="tiny muted">Showing the first {MAX_ROWS} of {results.length} rows.</p>
          )}
        </div>
      )}

      {!results && (
        <div className="hint-row">
          <Icon name="info" size={14} />
          <span>
            A sweep asks the server for every combination in the grid, so you see the network's real
            behaviour — not an approximation.
          </span>
        </div>
      )}
    </div>
  );
}
