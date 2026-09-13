import { useRef, useState } from 'react';
import Icon from '../Icon';
import { Badge, Button, NumberInput, Select } from '../ui';
import { ACTIVATIONS, LAYER_BY_TYPE, layerLabel, layerSummary, makeLayer } from '../../lib/layers';
import { LAYER_TYPE_COLORS } from '../../lib/colors';

/**
 * LayerStack — the editable column of layers between input and output.
 * Inline editing, drag-to-reorder, duplicate and remove.
 */
export default function LayerStack({ layers, inputs, outputs, inputLabels, outputLabels, onChange }) {
  const [dragIndex, setDragIndex] = useState(null);
  const [overIndex, setOverIndex] = useState(null);
  const listRef = useRef(null);

  const update = (next) => onChange(next);

  const patchLayer = (index, patch) => {
    update(layers.map((l, i) => (i === index ? { ...l, ...patch } : l)));
  };

  const removeLayer = (index) => update(layers.filter((_, i) => i !== index));

  const duplicateLayer = (index) => {
    const copy = { ...layers[index], id: undefined };
    const next = [...layers];
    next.splice(index + 1, 0, makeLayer(copy.type, copy));
    update(next);
  };

  const move = (from, to) => {
    if (to < 0 || to >= layers.length || from === to) return;
    const next = [...layers];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    update(next);
  };

  return (
    <div className="stack">
      <EndpointRow
        kind="input"
        title="Input"
        count={inputs}
        labels={inputLabels}
        hint="Fixed by the task"
      />

      <div className="stack__connector" aria-hidden="true" />

      <div className="stack__layers" ref={listRef}>
        {layers.length === 0 && (
          <div className="stack__empty">
            <Icon name="layers" size={15} />
            <span>No hidden layers — this is a linear model.</span>
          </div>
        )}

        {layers.map((layer, index) => {
          const spec = LAYER_BY_TYPE[layer.type];
          const color = LAYER_TYPE_COLORS[layer.type] || 'var(--accent)';
          return (
            <div
              key={layer.id || index}
              className="layerrow"
              draggable
              data-dragging={dragIndex === index ? 'true' : 'false'}
              data-dropbefore={overIndex === index && dragIndex !== null && dragIndex > index ? 'true' : 'false'}
              data-dropafter={
                overIndex === index && dragIndex !== null && dragIndex < index ? 'true' : 'false'
              }
              onDragStart={(e) => {
                setDragIndex(index);
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', String(index));
              }}
              onDragOver={(e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                if (overIndex !== index) setOverIndex(index);
              }}
              onDragLeave={() => setOverIndex((v) => (v === index ? null : v))}
              onDrop={(e) => {
                e.preventDefault();
                const from = Number(e.dataTransfer.getData('text/plain'));
                move(Number.isNaN(from) ? dragIndex : from, index);
                setDragIndex(null);
                setOverIndex(null);
              }}
              onDragEnd={() => {
                setDragIndex(null);
                setOverIndex(null);
              }}
              style={{ '--layer-color': color }}
            >
              <span className="layerrow__grip" title="Drag to reorder">
                <Icon name="grip" size={13} />
              </span>

              <span className="layerrow__icon">
                <Icon name={spec?.icon || 'layers'} size={14} />
              </span>

              <span className="layerrow__main">
                <span className="layerrow__title">
                  {layerLabel(layer.type)}
                  <em className="layerrow__idx">L{index + 1}</em>
                </span>
                <span className="layerrow__sub mono">{layerSummary(layer)}</span>
              </span>

              <span className="layerrow__fields">
                {layer.type === 'dense' && (
                  <>
                    <NumberInput
                      value={layer.neurons}
                      min={1}
                      max={512}
                      onChange={(v) => patchLayer(index, { neurons: Math.max(1, Number(v) || 1) })}
                      style={{ width: 52, height: 26 }}
                      aria-label="Neurons"
                    />
                    <Select
                      value={layer.activation || 'tanh'}
                      onChange={(v) => patchLayer(index, { activation: v })}
                      options={ACTIVATIONS.map((a) => ({ value: a.key, label: a.label }))}
                      style={{ width: 88, height: 26 }}
                      aria-label="Activation"
                    />
                  </>
                )}
                {layer.type === 'dropout' && (
                  <NumberInput
                    value={layer.rate}
                    min={0}
                    max={0.9}
                    step={0.05}
                    onChange={(v) => patchLayer(index, { rate: Number(v) })}
                    style={{ width: 62, height: 26 }}
                    aria-label="Dropout rate"
                  />
                )}
                {layer.type === 'conv2d' && (
                  <>
                    <NumberInput
                      value={layer.out_channels}
                      min={1}
                      max={256}
                      onChange={(v) => patchLayer(index, { out_channels: Number(v) })}
                      style={{ width: 52, height: 26 }}
                      aria-label="Filters"
                    />
                    <NumberInput
                      value={layer.kernel_size}
                      min={1}
                      max={7}
                      onChange={(v) => patchLayer(index, { kernel_size: Number(v) })}
                      style={{ width: 46, height: 26 }}
                      aria-label="Kernel size"
                    />
                  </>
                )}
                {(layer.type === 'lstm' || layer.type === 'simple_rnn') && (
                  <NumberInput
                    value={layer.hidden_size}
                    min={8}
                    max={512}
                    onChange={(v) => patchLayer(index, { hidden_size: Number(v) })}
                    style={{ width: 58, height: 26 }}
                    aria-label="Hidden size"
                  />
                )}
                {!['dense', 'dropout', 'conv2d', 'lstm', 'simple_rnn'].includes(layer.type) && (
                  <Badge mono>{layerSummary(layer) || 'no params'}</Badge>
                )}
              </span>

              <span className="layerrow__actions">
                <button
                  className="tool-btn"
                  title="Move up"
                  disabled={index === 0}
                  onClick={() => move(index, index - 1)}
                >
                  <Icon name="chevronUp" size={13} />
                </button>
                <button
                  className="tool-btn"
                  title="Move down"
                  disabled={index === layers.length - 1}
                  onClick={() => move(index, index + 1)}
                >
                  <Icon name="chevronDown" size={13} />
                </button>
                <button className="tool-btn" title="Duplicate" onClick={() => duplicateLayer(index)}>
                  <Icon name="copy" size={13} />
                </button>
                <button
                  className="tool-btn"
                  title="Remove layer"
                  onClick={() => removeLayer(index)}
                  style={{ color: 'var(--neg)' }}
                >
                  <Icon name="trash" size={13} />
                </button>
              </span>
            </div>
          );
        })}
      </div>

      <div className="stack__connector" aria-hidden="true" />

      <EndpointRow
        kind="output"
        title="Output"
        count={outputs}
        labels={outputLabels}
        hint="Fixed by the task"
      />
    </div>
  );
}

function EndpointRow({ kind, title, count, labels, hint }) {
  const preview = (labels || []).slice(0, 6);
  return (
    <div className={`endpoint endpoint--${kind}`}>
      <span className="endpoint__icon">
        <Icon name={kind === 'input' ? 'arrowRight' : 'target'} size={14} />
      </span>
      <span className="endpoint__main">
        <span className="endpoint__title">
          {title}
          <em className="mono">{count} neuron{count === 1 ? '' : 's'}</em>
        </span>
        <span className="endpoint__sub tiny muted truncate">
          {preview.length ? preview.join(' · ') : hint}
          {labels && labels.length > preview.length ? ` +${labels.length - preview.length}` : ''}
        </span>
      </span>
    </div>
  );
}
