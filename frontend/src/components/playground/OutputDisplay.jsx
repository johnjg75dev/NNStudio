import { useEffect, useRef, useState } from 'react';
import Icon from '../Icon';
import { Badge, Button, EmptyState, Field, Select } from '../ui';
import { PixelPreview } from '../shared/PixelPreview';
import { SevenSegment, fourBitToHex } from '../shared/SevenSegment';
import { useIoShape } from '../../lib/ioShape';
import { useSession, useSessionActions } from '../../state/SessionContext';
import { fmtNum } from '../../lib/format';
import { argmax, matchesTarget } from '../../lib/samples';

const MAX_RUNS = 12;

/** OutputDisplay — the network's answer for the current input vector. */
export default function OutputDisplay() {
  const actions = useSessionActions();
  const snapshot = useSession((s) => s.snapshot);
  const test = useSession((s) => s.test);
  const busy = useSession((s) => s.busy);
  const io = useIoShape();

  const [runs, setRuns] = useState([]);
  const lastSeen = useRef(null);

  const func = snapshot?.func || null;
  const isSeg7 = func?.key === 'seg7';
  const topology = snapshot?.topology || [];
  const columns = topology.map((n, i) => ({ value: String(i), label: `Column ${i} · ${n} neurons` }));

  // Collect a short trail of the predictions made anywhere in the app.
  useEffect(() => {
    if (!test.output) return;
    const key = `${JSON.stringify(test.x || test.values)}|${JSON.stringify(test.output)}`;
    if (key === lastSeen.current) return;
    lastSeen.current = key;
    setRuns((prev) =>
      [{ x: test.x || test.values, out: test.output, expected: test.expected, at: Date.now(), source: test.source }, ...prev].slice(0, MAX_RUNS),
    );
  }, [test.output, test.x, test.values, test.expected, test.source]);

  if (!snapshot?.built) {
    return (
      <EmptyState
        icon="bolt"
        title="Nothing to run yet"
        message="Build a network on the Train page — the playground then lets you push any input through it."
      />
    );
  }

  const output = test.output || null;
  const expected = test.expected || null;
  const correct = output && expected ? matchesTarget(output, expected) : null;
  const winner = output && output.length > 1 ? argmax(output) : null;
  const labels = func?.output_labels || (output || []).map((_, i) => `y${i}`);

  const predict = () => actions.predict({ x: test.values, source: 'playground' });

  return (
    <div className="output">
      <div className="output__bar">
        <Button size="sm" variant="primary" icon="bolt" loading={busy} onClick={predict}>
          Run forward pass
        </Button>
        <div className="row" style={{ gap: 6, marginLeft: 'auto' }}>
          <Field label="from" inline className="field--tight">
            <Select
              value={String(test.startLayer ?? 0)}
              onChange={(v) => actions.predict({ x: test.values, startLayer: Number(v), endLayer: test.endLayer, source: 'playground' })}
              options={columns}
              className="select--sm"
            />
          </Field>
          <Field label="to" inline className="field--tight">
            <Select
              value={test.endLayer === null || test.endLayer === undefined ? 'end' : String(test.endLayer)}
              onChange={(v) =>
                actions.predict({
                  x: test.values,
                  startLayer: test.startLayer ?? 0,
                  endLayer: v === 'end' ? null : Number(v),
                  source: 'playground',
                })
              }
              options={[...columns, { value: 'end', label: 'output' }]}
              className="select--sm"
            />
          </Field>
        </div>
      </div>

      {!output ? (
        <div className="hint-row">
          <Icon name="info" size={14} />
          <span>Set the inputs on the left, then run a forward pass to see what the network says.</span>
        </div>
      ) : isSeg7 ? (
        <div className="output__seg">
          <SevenSegment values={output} size={96} />
          <div className="col" style={{ gap: 6 }}>
            <Badge tone="violet">digit {fourBitToHex(test.x || test.values)}</Badge>
            <div className="row wrap" style={{ gap: 4 }}>
              {output.map((v, i) => (
                <span key={i} className={`io__chip mono ${Number(v) > 0.5 ? 'pos' : ''}`}>
                  s{i} {fmtNum(v)}
                </span>
              ))}
            </div>
          </div>
        </div>
      ) : io.output && io.isImage ? (
        <div className="output__image">
          <PixelPreview data={output} shape={io.output} small={false} max={160} />
          {expected && (
            <>
              <span className="tiny muted">expected</span>
              <PixelPreview data={expected} shape={io.output} small={false} max={160} />
            </>
          )}
        </div>
      ) : (
        <div className="output__values">
          {output.map((v, i) => {
            const n = Number(v) || 0;
            const isWin = winner === i;
            return (
              <div key={i} className={`out-row ${isWin ? 'out-row--win' : ''}`}>
                <span className="out-row__label tiny truncate" title={labels[i] || `y${i}`}>
                  {labels[i] || `y${i}`}
                </span>
                <span className="out-row__bar">
                  <span
                    className={`out-row__fill ${n >= 0 ? 'pos' : 'neg'}`}
                    style={{ width: `${Math.min(100, Math.abs(n) * 100)}%` }}
                  />
                </span>
                <span className="out-row__val mono">{fmtNum(n)}</span>
                {expected && (
                  <span className="out-row__exp mono muted" title={`expected ${fmtNum(expected[i])}`}>
                    {fmtNum(expected[i])}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {correct !== null && (
        <div className={`banner ${correct ? 'banner--pos' : 'banner--warn'}`}>
          <Icon name={correct ? 'check' : 'alert'} size={14} />
          <span>
            {correct
              ? 'Matches the expected output for this input.'
              : 'Differs from the expected output — train more or adjust the setup.'}
          </span>
        </div>
      )}

      {runs.length > 1 && (
        <section className="output__runs">
          <header className="sec-head">
            <h3>Recent runs</h3>
            <Button size="xs" variant="ghost" icon="trash" onClick={() => setRuns([])}>
              Clear
            </Button>
          </header>
          <ul className="runs">
            {runs.map((r, i) => (
              <li key={i} className="runs__item">
                <button
                  type="button"
                  className="runs__load"
                  title="Load this input again"
                  onClick={() => actions.predict({ x: r.x, y: r.expected, source: 'playground' })}
                >
                  <span className="mono xs truncate">{(r.x || []).map(fmtNum).join(' · ')}</span>
                  <Icon name="arrowRight" size={11} />
                </button>
                <span className="mono xs muted truncate" title={(r.out || []).map(fmtNum).join(', ')}>
                  {(r.out || []).map(fmtNum).join(' · ')}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
