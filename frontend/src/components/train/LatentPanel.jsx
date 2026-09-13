import { useEffect, useMemo, useState } from 'react';
import Icon from '../Icon';
import SweepChart from '../canvas/SweepChart';
import { Badge, Button, EmptyState, Field, Select, Slider } from '../ui';
import { useSession, useSessionActions, useSessionStore } from '../../state/SessionContext';
import { fmtNum, truncateArray } from '../../lib/format';

/** Latent explorer: force one hidden neuron and watch the outputs respond. */
export default function LatentPanel({ initialNode }) {
  const store = useSessionStore();
  const actions = useSessionActions();

  const snapshot = useSession((s) => s.snapshot);
  const latent = useSession((s) => s.latent);
  const test = useSession((s) => s.test);
  const samples = useSession((s) => s.samples);
  const busy = useSession((s) => s.busy);

  const hidden = useMemo(
    () => (snapshot?.layers || []).filter((l) => !l.is_output).map((l) => ({ column: l.index + 1, n: l.n_out })),
    [snapshot],
  );

  const [layer, setLayer] = useState(1);
  const [node, setNode] = useState(0);
  const [value, setValue] = useState(0);
  const [min, setMin] = useState(-2);
  const [max, setMax] = useState(2);
  const [steps, setSteps] = useState(24);
  const [base, setBase] = useState('zeros');

  // Adopt a node handed over from the inspector.
  useEffect(() => {
    if (!initialNode) return;
    setLayer(initialNode.layer);
    setNode(initialNode.idx);
  }, [initialNode]);

  useEffect(() => {
    if (hidden.length && !hidden.some((h) => h.column === layer)) {
      setLayer(hidden[0].column);
      setNode(0);
    }
  }, [hidden, layer]);

  const current = hidden.find((h) => h.column === layer);
  const nodeCount = current?.n || 0;
  const func = snapshot?.func;

  const baseInput = useMemo(() => {
    if (base === 'zeros') return (test.values || []).map(() => 0);
    if (base === 'current') return test.values || [];
    const idx = Number(String(base).replace('sample-', ''));
    return samples[idx]?.x || test.values || [];
  }, [base, test.values, samples]);

  if (!snapshot?.built) {
    return (
      <EmptyState
        icon="wave"
        title="Build a network first"
        message="The latent explorer overrides one hidden neuron at a time, so it needs a built network to poke at."
      />
    );
  }

  if (!hidden.length) {
    return (
      <EmptyState
        icon="wave"
        title="No hidden layers"
        message="Add at least one hidden layer in the setup panel — an input→output wire has no latent space to explore."
      />
    );
  }

  async function apply(nextValue = value) {
    store.setTestValues(baseInput);
    await actions.latentSweep({ layer, node, value: nextValue });
  }

  async function sweepRange() {
    store.setTestValues(baseInput);
    const step = (max - min) / Math.max(2, steps);
    await actions.latentSweep({ layer, node, value, range: [min, max, step] });
  }

  return (
    <div className="latent">
      <div className="row wrap" style={{ gap: 10 }}>
        <Field label="Hidden layer" className="grow">
          <Select
            value={String(layer)}
            onChange={(v) => {
              setLayer(Number(v));
              setNode(0);
            }}
            options={hidden.map((h) => ({ value: String(h.column), label: `Column ${h.column} · ${h.n} neurons` }))}
          />
        </Field>
        <Field label="Neuron" style={{ width: 120 }}>
          <Select
            value={String(node)}
            onChange={(v) => setNode(Number(v))}
            options={Array.from({ length: nodeCount }, (_, i) => ({ value: String(i), label: `n${i}` }))}
          />
        </Field>
      </div>

      <Field label="Base input" hint="What the rest of the network sees while this neuron is forced." className="mt-8">
        <Select
          value={base}
          onChange={setBase}
          options={[
            { value: 'zeros', label: 'All zeros' },
            { value: 'current', label: `Current test input [${truncateArray(test.values, 4, (v) => fmtNum(v)).join(' ')}]` },
            ...samples.slice(0, 8).map((s, i) => ({
              value: `sample-${i}`,
              label: `Training sample #${i} — ${s.label || truncateArray(s.x, 4).join(' ')}`,
            })),
          ]}
        />
      </Field>

      <Slider
        className="mt-8"
        label="Forced value"
        value={value}
        min={-3}
        max={3}
        step={0.05}
        format={fmtNum}
        onChange={setValue}
        onCommit={(v) => apply(v)}
      />

      <div className="row wrap" style={{ gap: 6, margin: '10px 0' }}>
        <Button size="sm" variant="primary" icon="bolt" loading={busy} onClick={() => apply()}>
          Apply override
        </Button>
        <Button size="sm" icon="wave" loading={busy} onClick={sweepRange}>
          Sweep range
        </Button>
        {latent?.data && (
          <Button size="sm" variant="ghost" icon="trash" onClick={() => store.clearLatent()}>
            Clear
          </Button>
        )}
      </div>

      <div className="row wrap" style={{ gap: 8 }}>
        <Field label="Sweep from" style={{ width: 90 }}>
          <input className="input input--sm mono" type="number" step="0.5" value={min} onChange={(e) => setMin(Number(e.target.value))} />
        </Field>
        <Field label="to" style={{ width: 90 }}>
          <input className="input input--sm mono" type="number" step="0.5" value={max} onChange={(e) => setMax(Number(e.target.value))} />
        </Field>
        <Field label="points" style={{ width: 90 }}>
          <input className="input input--sm mono" type="number" min="4" max="120" step="2" value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
        </Field>
      </div>

      {latent?.output && (
        <div className="latent__output">
          <Badge tone="violet">output with n{latent.node} forced to {fmtNum(latent.value)}</Badge>
          <div className="row wrap" style={{ gap: 4, marginTop: 6 }}>
            {latent.output.map((v, i) => (
              <span key={i} className="io__chip mono">
                {func?.output_labels?.[i] ? `${func.output_labels[i]} ` : `y${i} `}
                {fmtNum(v)}
              </span>
            ))}
          </div>
        </div>
      )}

      {latent?.data && latent.data.length > 1 ? (
        <div className="latent__sweep">
          <header className="sec-head">
            <h3>Output vs. forced neuron</h3>
            <Badge mono>
              {fmtNum(latent.data[0].val)} → {fmtNum(latent.data[latent.data.length - 1].val)}
            </Badge>
          </header>
          <SweepChart data={latent.data} height={180} />
          <p className="tiny muted mt-8">
            Each curve is one network output while column {layer} · neuron {node} is forced across the
            range. A flat line means this neuron barely matters for the task right now.
          </p>
        </div>
      ) : (
        <div className="hint-row">
          <Icon name="info" size={14} />
          <span>
            Drag the slider and release to apply a single override, or sweep a range to plot every
            output against it.
          </span>
        </div>
      )}
    </div>
  );
}
