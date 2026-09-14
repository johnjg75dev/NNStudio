import { useEffect, useMemo, useState } from 'react';
import Icon from '../Icon';
import PixelCanvas from '../canvas/PixelCanvas';
import { Badge, Button, Field, NumberInput, Segmented, Select, Slider } from '../ui';
import { useIoShape } from '../../lib/ioShape';
import { useSession, useSessionActions } from '../../state/SessionContext';
import { fmtNum } from '../../lib/format';

/**
 * InputControls — numeric sliders, a pixel drawing pad, or quick sample loading.
 * Values live in the session store so every panel sees the same input vector.
 */
export default function InputControls({ mode, onModeChange }) {
  const actions = useSessionActions();
  const snapshot = useSession((s) => s.snapshot);
  const test = useSession((s) => s.test);
  const samples = useSession((s) => s.samples);
  const sweep = useSession((s) => s.sweep);
  const busy = useSession((s) => s.busy);
  const io = useIoShape();

  const [autoRun, setAutoRun] = useState(true);

  const func = snapshot?.func || null;
  const labels = func?.input_labels || test.values.map((_, i) => `x${i}`);
  const values = test.values || [];

  // Auto-fallback if 'draw' mode was active but current task is not an image
  useEffect(() => {
    if (mode === 'draw' && !io.input) {
      onModeChange?.('numbers');
    }
  }, [mode, io.input, onModeChange]);

  // Debounced live prediction on input change
  useEffect(() => {
    if (!autoRun || !snapshot?.built || !values.length) return;
    const timer = setTimeout(() => {
      actions.predict({ x: values, source: 'playground' });
    }, 120);
    return () => clearTimeout(timer);
  }, [values, autoRun, snapshot?.built]);

  const bounds = useMemo(
    () =>
      labels.map((_, i) => {
        const r = sweep?.ranges?.[i];
        if (r && (r.min !== r.max)) return { min: Number(r.min), max: Number(r.max), step: Number(r.step) || 0.05 };
        const col = samples.map((s) => Number(s.x?.[i]) || 0);
        if (col.length) {
          const min = Math.min(...col);
          const max = Math.max(...col);
          if (max > min) return { min, max, step: (max - min) / 20 };
        }
        return { min: 0, max: 1, step: 0.05 };
      }),
    [labels, samples, sweep],
  );

  if (!snapshot?.built) return null;

  const setValue = (i, v) => {
    const next = [...values];
    next[i] = Number(v) || 0;
    actions.setTestValues(next);
  };

  const modes = [
    { value: 'numbers', label: 'Numbers', icon: 'sliders' },
    ...(io.input ? [{ value: 'draw', label: 'Draw', icon: 'image' }] : []),
    { value: 'samples', label: 'Samples', icon: 'database' },
  ];

  return (
    <div className="inputs">
      <div className="row wrap" style={{ gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <Segmented value={mode} onChange={onModeChange} options={modes} />
        <Button
          size="sm"
          variant="primary"
          icon="bolt"
          loading={busy}
          onClick={() => actions.predict({ x: values, source: 'playground' })}
          title="Run forward pass through the network"
        >
          Predict
        </Button>
        <button
          type="button"
          className={`chip-toggle ${autoRun ? 'chip-toggle--active' : ''}`}
          onClick={() => setAutoRun(!autoRun)}
          title="Toggle live prediction as you move sliders or draw"
        >
          <Icon name={autoRun ? 'check' : 'close'} size={11} />
          <span>Live pass</span>
        </button>
        <Badge mono style={{ marginLeft: 'auto' }}>
          {values.length} inputs
        </Badge>
      </div>

      {mode === 'numbers' && (
        <>
          <div className={`inputs__grid ${labels.length > 8 ? 'inputs__grid--dense' : ''}`}>
            {labels.map((label, i) => (
              <div key={`${label}-${i}`} className="input-row">
                <div className="row row--between">
                  <span className="input-row__label tiny">{label}</span>
                  <NumberInput
                    className="input--xs mono"
                    value={Number(values[i] ?? 0)}
                    step={bounds[i].step}
                    onChange={(v) => setValue(i, v)}
                    style={{ width: 68 }}
                  />
                </div>
                <Slider
                  value={Number(values[i] ?? 0)}
                  min={bounds[i].min}
                  max={bounds[i].max}
                  step={bounds[i].step}
                  onChange={(v) => setValue(i, v)}
                />
              </div>
            ))}
          </div>
          <div className="row wrap" style={{ gap: 6, marginTop: 12 }}>
            <Button size="sm" icon="dice" onClick={() => actions.setTestValues(labels.map((_, i) => rand(bounds[i])))}>
              Randomise
            </Button>
            <Button size="sm" variant="ghost" icon="reset" onClick={() => actions.setTestValues(labels.map(() => 0))}>
              Zero
            </Button>
            <Button
              size="sm"
              variant="ghost"
              icon="shuffle"
              onClick={() => actions.setTestValues(labels.map((_, i) => (Number(values[i] ?? 0) > 0.5 ? 0 : 1)))}
            >
              Flip binary
            </Button>
          </div>
        </>
      )}

      {mode === 'draw' && io.input && (
        <div className="inputs__draw">
          <PixelCanvas
            width={io.input[0]}
            height={io.input[1]}
            channels={io.input[2] || 1}
            values={values}
            zoom={Math.max(6, Math.min(18, Math.floor(240 / io.input[0])))}
            onChange={(next) => actions.setTestValues(next)}
          />
          <div className="row wrap" style={{ gap: 6, marginTop: 10 }}>
            <Button size="sm" variant="ghost" icon="trash" onClick={() => actions.setTestValues(values.map(() => 0))}>
              Clear canvas
            </Button>
            <Button size="sm" variant="ghost" icon="dice" onClick={() => actions.setTestValues(values.map(() => Math.random()))}>
              Noise
            </Button>
            <span className="tiny muted">
              {io.input[0]}×{io.input[1]}×{io.input[2] || 1} · painted pixels become inputs 0…1
            </span>
          </div>
        </div>
      )}

      {mode === 'samples' && (
        <div className="inputs__samples">
          {!samples.length ? (
            <p className="tiny muted">No evaluated samples yet — run a training step or re-evaluate.</p>
          ) : (
            <>
              <Field label="Load a training sample as input">
                <Select
                  value=""
                  onChange={(v) => {
                    const s = samples[Number(v)];
                    if (s) actions.predict({ x: s.x, y: s.y, source: 'playground' });
                  }}
                  options={samples.slice(0, 60).map((s, i) => ({
                    value: String(i),
                    label: `#${i} — ${s.label || `x=[${(s.x || []).slice(0, 4).map(fmtNum).join(', ')}]`}`,
                  }))}
                />
              </Field>
              <div className="sample-chips">
                {samples.slice(0, 24).map((s, i) => (
                  <button
                    key={i}
                    type="button"
                    className={`sample-chip ${s.correct ? '' : 'sample-chip--wrong'}`}
                    title={(s.x || []).map(fmtNum).join(', ')}
                    onClick={() => actions.predict({ x: s.x, y: s.y, source: 'playground' })}
                  >
                    <Icon name={s.correct ? 'check' : 'close'} size={11} />
                    <span className="mono xs">{s.label || `#${i}`}</span>
                  </button>
                ))}
              </div>
              <p className="tiny muted">
                Chips are the evaluated training samples — ✓ where the network is already right. Click one
                to push its input through the network here.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function rand({ min, max }) {
  const v = min + Math.random() * (max - min);
  return Math.round(v * 1000) / 1000;
}
