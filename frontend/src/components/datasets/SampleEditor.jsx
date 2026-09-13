import { useMemo, useState } from 'react';
import Icon from '../Icon';
import PixelCanvas from '../canvas/PixelCanvas';
import ScatterPlot from './ScatterPlot';
import { Badge, Button, EmptyState, NumberInput, Segmented, Select, Slider, Tabs } from '../ui';
import { fmtNum } from '../../lib/format';

const MAX_COLS = 12;

/**
 * SampleEditor — browse and edit a dataset's samples.
 * Tabular data gets a spreadsheet, image data gets the pixel editor, and
 * two-input datasets also get a scatter view.
 */
export default function SampleEditor({ dataset, readOnly = false, data, onChange }) {
  const samples = Array.isArray(data) ? data : [];
  const numInputs = Number(dataset?.num_inputs) || samples[0]?.x?.length || 0;
  const numOutputs = Number(dataset?.num_outputs) || samples[0]?.y?.length || 0;
  const isImage = Boolean(dataset?.width && dataset?.height);
  const inLabels = dataset?.input_labels?.length ? dataset.input_labels : null;
  const outLabels = dataset?.output_labels?.length ? dataset.output_labels : null;

  const [tab, setTab] = useState(isImage ? 'image' : 'table');
  const [cursor, setCursor] = useState(0);
  const [brush, setBrush] = useState(1);
  const [json, setJson] = useState('');
  const [jsonError, setJsonError] = useState(null);

  const idx = Math.min(cursor, Math.max(0, samples.length - 1));
  const sample = samples[idx] || null;

  const stats = useMemo(() => {
    if (!samples.length) return null;
    const cols = numInputs;
    const mins = new Array(cols).fill(Infinity);
    const maxs = new Array(cols).fill(-Infinity);
    samples.forEach((s) => {
      for (let i = 0; i < cols; i += 1) {
        const v = Number(s.x?.[i]) || 0;
        if (v < mins[i]) mins[i] = v;
        if (v > maxs[i]) maxs[i] = v;
      }
    });
    return { mins, maxs };
  }, [samples, numInputs]);

  const tabs = [
    { value: 'table', label: 'Table', icon: 'table' },
    ...(isImage ? [{ value: 'image', label: 'Pixels', icon: 'image' }] : []),
    ...(numInputs === 2 ? [{ value: 'scatter', label: 'Scatter', icon: 'chart' }] : []),
    { value: 'json', label: 'JSON', icon: 'code' },
  ];

  function update(i, patch) {
    if (readOnly) return;
    const next = samples.map((s, j) => (j === i ? { ...s, ...patch } : s));
    onChange(next);
  }

  function setX(i, dim, value) {
    const s = samples[i];
    if (!s) return;
    const x = [...(s.x || [])];
    x[dim] = Number(value) || 0;
    update(i, { x });
  }

  function setY(i, dim, value) {
    const s = samples[i];
    if (!s) return;
    const y = [...(s.y || [])];
    y[dim] = Number(value) || 0;
    update(i, { y });
  }

  function addSample() {
    if (readOnly) return;
    const x = new Array(numInputs).fill(0).map(() => Math.round(Math.random() * 100) / 100);
    const y = numOutputs > 1 ? oneHot(numOutputs, samples.length % numOutputs) : new Array(Math.max(1, numOutputs)).fill(0);
    onChange([...samples, { x, y: dataset?.is_input_only ? [] : y }]);
    setCursor(samples.length);
  }

  function removeSample(i) {
    if (readOnly) return;
    onChange(samples.filter((_, j) => j !== i));
    setCursor((c) => Math.max(0, Math.min(c, samples.length - 2)));
  }

  function duplicateSample(i) {
    if (readOnly) return;
    const copy = JSON.parse(JSON.stringify(samples[i]));
    const next = [...samples];
    next.splice(i + 1, 0, copy);
    onChange(next);
    setCursor(i + 1);
  }

  if (!samples.length) {
    return (
      <EmptyState
        icon="database"
        title="This dataset has no samples"
        message={
          readOnly
            ? 'Predefined image datasets download their samples on demand — press Download on the dataset card.'
            : 'Add a few samples to start training on it.'
        }
        action={readOnly ? null : <Button icon="plus" variant="primary" onClick={addSample}>Add a sample</Button>}
      />
    );
  }

  return (
    <div className="sample-editor">
      <Tabs value={tab} onChange={setTab} tabs={tabs} />

      <div className="sample-editor__bar">
        <Badge mono>{samples.length} samples</Badge>
        <Badge mono>
          {numInputs} in{numOutputs ? ` · ${numOutputs} out` : ''}
        </Badge>
        {stats && (
          <Badge mono title="Range across all samples">
            x ∈ [{fmtNum(Math.min(...stats.mins))}, {fmtNum(Math.max(...stats.maxs))}]
          </Badge>
        )}
        {readOnly && <Badge tone="warn">read-only</Badge>}
        {!readOnly && (
          <div className="row" style={{ gap: 6, marginLeft: 'auto' }}>
            <Button size="sm" icon="plus" onClick={addSample}>
              Add sample
            </Button>
            <Button size="sm" variant="ghost" icon="dice" onClick={() => onChange(samples.map((s) => ({ ...s, x: s.x.map(() => Math.round(Math.random() * 1000) / 1000) })))}>
              Shuffle inputs
            </Button>
          </div>
        )}
      </div>

      {tab === 'table' && (
        <div className="table-scroll">
          <table className="table table--samples">
            <thead>
              <tr>
                <th>#</th>
                {Array.from({ length: Math.min(numInputs, MAX_COLS) }, (_, i) => (
                  <th key={`x${i}`} className="mono">
                    {inLabels?.[i] || `x${i}`}
                  </th>
                ))}
                {numInputs > MAX_COLS && <th className="mono muted">+{numInputs - MAX_COLS}</th>}
                {!dataset?.is_input_only &&
                  Array.from({ length: Math.min(numOutputs, 6) }, (_, i) => (
                    <th key={`y${i}`} className="mono th--out">
                      {outLabels?.[i] || `y${i}`}
                    </th>
                  ))}
                <th aria-label="actions" />
              </tr>
            </thead>
            <tbody>
              {samples.map((s, i) => (
                <tr key={i} data-active={i === idx ? 'true' : 'false'} onClick={() => setCursor(i)}>
                  <td className="mono muted">{i}</td>
                  {(s.x || []).slice(0, MAX_COLS).map((v, d) => (
                    <td key={d} className="cell--num">
                      {readOnly ? (
                        <span className="mono tiny">{fmtNum(v)}</span>
                      ) : (
                        <NumberInput
                          className="input--xs mono"
                          value={Number(v) || 0}
                          step={0.01}
                          onChange={(nv) => setX(i, d, nv)}
                        />
                      )}
                    </td>
                  ))}
                  {numInputs > MAX_COLS && (
                    <td className="mono tiny muted" title={(s.x || []).slice(MAX_COLS).map(fmtNum).join(', ')}>
                      …
                    </td>
                  )}
                  {!dataset?.is_input_only &&
                    (s.y || []).slice(0, 6).map((v, d) => (
                      <td key={d} className="cell--num cell--out">
                        {readOnly ? (
                          <span className="mono tiny">{fmtNum(v)}</span>
                        ) : (
                          <NumberInput
                            className="input--xs mono"
                            value={Number(v) || 0}
                            step={1}
                            onChange={(nv) => setY(i, d, nv)}
                          />
                        )}
                      </td>
                    ))}
                  <td>
                    {!readOnly && (
                      <div className="row" style={{ gap: 2 }}>
                        <Button size="xs" variant="ghost" iconOnly title="Duplicate" onClick={(e) => { e.stopPropagation(); duplicateSample(i); }}>
                          <Icon name="copy" size={12} />
                        </Button>
                        <Button size="xs" variant="ghost" iconOnly title="Delete sample" onClick={(e) => { e.stopPropagation(); removeSample(i); }}>
                          <Icon name="trash" size={12} />
                        </Button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'image' && isImage && sample && (
        <div className="pixel-editor">
          <div className="pixel-editor__stage">
            <PixelCanvas
              width={dataset.width}
              height={dataset.height}
              channels={dataset.channels || 1}
              values={sample.x || []}
              zoom={Math.max(6, Math.min(20, Math.floor(320 / dataset.width)))}
              readOnly={readOnly}
              brushSize={brush}
              onChange={(next) => update(idx, { x: next })}
            />
          </div>
          <div className="pixel-editor__side">
            <div className="row" style={{ gap: 6 }}>
              <Button size="sm" variant="ghost" iconOnly disabled={idx === 0} onClick={() => setCursor(idx - 1)}>
                <Icon name="chevronLeft" />
              </Button>
              <Select
                value={String(idx)}
                onChange={(v) => setCursor(Number(v))}
                options={samples.map((_, i) => ({ value: String(i), label: `Sample ${i}` }))}
                className="grow"
              />
              <Button size="sm" variant="ghost" iconOnly disabled={idx >= samples.length - 1} onClick={() => setCursor(idx + 1)}>
                <Icon name="chevronRight" />
              </Button>
            </div>

            {!readOnly && (
              <Slider label="Brush size" value={brush} min={1} max={4} step={1} onChange={setBrush} />
            )}

            {!dataset?.is_input_only && (
              <div className="field">
                <span className="field__label">Target</span>
                <div className="row wrap" style={{ gap: 4 }}>
                  {(sample.y || []).map((v, d) => (
                    <label key={d} className="chip-toggle">
                      <input
                        type="radio"
                        name={`target-${idx}`}
                        checked={argmax(sample.y) === d}
                        disabled={readOnly}
                        onChange={() => update(idx, { y: oneHot(sample.y.length, d) })}
                      />
                      <span className="mono xs">{outLabels?.[d] || `y${d}`}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <Histogram values={sample.x || []} />

            {!readOnly && (
              <div className="row wrap" style={{ gap: 6 }}>
                <Button size="sm" variant="ghost" icon="trash" onClick={() => update(idx, { x: (sample.x || []).map(() => 0) })}>
                  Clear
                </Button>
                <Button size="sm" variant="ghost" icon="dice" onClick={() => update(idx, { x: (sample.x || []).map(() => Math.random()) })}>
                  Noise
                </Button>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'scatter' && (
        <ScatterPlot samples={samples} onSelect={(s) => setCursor(samples.indexOf(s))} />
      )}

      {tab === 'json' && (
        <div className="json-editor">
          <p className="tiny muted">
            Paste an array of <span className="mono">{'{ x: [...], y: [...] }'}</span> objects. Applying
            replaces every sample{readOnly ? ' (disabled — this dataset is read-only)' : ''}.
          </p>
          <textarea
            className="textarea mono"
            rows={14}
            readOnly={readOnly}
            value={json}
            placeholder={JSON.stringify(samples.slice(0, 3), null, 2)}
            onChange={(e) => {
              setJson(e.target.value);
              setJsonError(null);
            }}
          />
          {jsonError && <div className="banner banner--neg"><Icon name="alert" size={14} /> {jsonError}</div>}
          {!readOnly && (
            <div className="row" style={{ gap: 6, marginTop: 8 }}>
              <Button
                size="sm"
                variant="primary"
                icon="check"
                onClick={() => {
                  try {
                    const parsed = JSON.parse(json);
                    if (!Array.isArray(parsed)) throw new Error('Expected an array of samples.');
                    const clean = parsed.map((s) => ({
                      x: (s.x || []).map((v) => Number(v) || 0),
                      y: (s.y || []).map((v) => Number(v) || 0),
                    }));
                    onChange(clean);
                    setJson('');
                    setJsonError(null);
                  } catch (e) {
                    setJsonError(e.message);
                  }
                }}
              >
                Apply JSON
              </Button>
              <Button
                size="sm"
                variant="ghost"
                icon="copy"
                onClick={() => setJson(JSON.stringify(samples, null, 2))}
              >
                Load current samples
              </Button>
            </div>
          )}
        </div>
      )}

      {tab === 'table' && !readOnly && (
        <div className="row wrap tiny muted" style={{ gap: 10, marginTop: 8 }}>
          <span>
            <Icon name="info" size={12} /> Click a row to make it the active sample for the pixel and
            scatter views.
          </span>
        </div>
      )}
    </div>
  );
}

function oneHot(n, i) {
  return Array.from({ length: n }, (_, k) => (k === i ? 1 : 0));
}

function argmax(values = []) {
  let best = 0;
  let bestV = -Infinity;
  values.forEach((v, i) => {
    const n = Number(v) || 0;
    if (n > bestV) {
      bestV = n;
      best = i;
    }
  });
  return best;
}

/** 32-bin histogram of the active sample's pixel values. */
function Histogram({ values, bins = 32 }) {
  const counts = useMemo(() => {
    const out = new Array(bins).fill(0);
    (values || []).forEach((v) => {
      const n = Math.max(0, Math.min(0.9999, Number(v) || 0));
      out[Math.floor(n * bins)] += 1;
    });
    return out;
  }, [values, bins]);
  const peak = Math.max(1, ...counts);
  return (
    <div className="histogram">
      <span className="field__label">Pixel histogram</span>
      <div className="histogram__bars">
        {counts.map((c, i) => (
          <span key={i} className="histogram__bar" style={{ height: `${(c / peak) * 100}%` }} title={`${i}/${bins}: ${c}`} />
        ))}
      </div>
      <div className="row row--between mono xs muted">
        <span>0.0</span>
        <span>{(values || []).length} px</span>
        <span>1.0</span>
      </div>
    </div>
  );
}
