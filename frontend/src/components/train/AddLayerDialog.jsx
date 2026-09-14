import { useEffect, useMemo, useState } from 'react';
import Icon from '../Icon';
import { Button, Checkbox, Field, Modal, NumberInput, Select, TextInput } from '../ui';
import { LAYER_BY_TYPE, LAYER_CATALOG } from '../../lib/layers';
import { LAYER_TYPE_COLORS } from '../../lib/colors';

/** "Add layer" dialog: catalogue on the left, configuration on the right. */
export default function AddLayerDialog({ open, onClose, onAdd, layerCount }) {
  const [type, setType] = useState('dense');
  const [values, setValues] = useState({});
  const [quantity, setQuantity] = useState(1);
  const [placement, setPlacement] = useState('bottom');
  const [index, setIndex] = useState(0);

  const spec = LAYER_BY_TYPE[type];

  useEffect(() => {
    if (!open) return;
    setType('dense');
    setValues(defaultsFor('dense'));
    setQuantity(1);
    setPlacement('bottom');
    setIndex(layerCount);
  }, [open, layerCount]);

  const positions = useMemo(
    () => [
      { value: 'bottom', label: 'At the end (before output)' },
      { value: 'top', label: 'At the start (after input)' },
      { value: 'after', label: 'At a specific position…' },
    ],
    [],
  );

  function defaultsFor(t) {
    const s = LAYER_BY_TYPE[t];
    const out = {};
    (s?.fields || []).forEach((f) => {
      out[f.id] = f.def;
    });
    return out;
  }

  function submit() {
    const n = Math.max(1, Math.min(10, Number(quantity) || 1));
    const at = placement === 'top' ? 0 : placement === 'after' ? Number(index) || 0 : layerCount;
    onAdd({ type, values, quantity: n, at });
    onClose();
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Add a layer"
      subtitle="Pick a layer type, configure it, then choose where it goes in the stack."
      icon="plus"
      size="wide"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" icon="plus" onClick={submit} disabled={!spec}>
            Add {quantity > 1 ? `${quantity} layers` : layerLabelSafe(type)}
          </Button>
        </>
      }
    >
      <div className="addlayer">
        <div className="addlayer__catalog">
          {LAYER_CATALOG.map((group) => (
            <section key={group.group}>
              <h4 className="section-title">{group.group}</h4>
              <div className="addlayer__grid">
                {group.items.map((item) => (
                  <button
                    key={item.type}
                    type="button"
                    className="layer-tile"
                    data-active={type === item.type ? 'true' : 'false'}
                    style={{ '--layer-color': LAYER_TYPE_COLORS[item.type] || 'var(--accent)' }}
                    onClick={() => {
                      setType(item.type);
                      setValues(defaultsFor(item.type));
                    }}
                  >
                    <Icon name={item.icon} size={17} />
                    <b>{item.name}</b>
                    <span>{item.blurb}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>

        <aside className="addlayer__config">
          <h4 className="section-title">Configuration</h4>
          {spec?.fields?.length ? (
            <div className="col" style={{ gap: 10 }}>
              {spec.fields.map((f) => (
                <Field key={f.id} label={f.label}>
                  {f.kind === 'select' ? (
                    <Select
                      value={values[f.id] ?? f.def}
                      onChange={(v) => setValues((s) => ({ ...s, [f.id]: v }))}
                      options={f.options.map((o) => ({ value: o.key, label: o.label }))}
                    />
                  ) : f.kind === 'checkbox' ? (
                    <Checkbox
                      checked={!!values[f.id]}
                      onChange={(v) => setValues((s) => ({ ...s, [f.id]: v }))}
                      label={f.label}
                    />
                  ) : f.kind === 'text' ? (
                    <TextInput
                      value={values[f.id] ?? f.def}
                      onChange={(v) => setValues((s) => ({ ...s, [f.id]: v }))}
                    />
                  ) : (
                    <NumberInput
                      value={values[f.id] ?? f.def}
                      min={f.min}
                      max={f.max}
                      step={f.step ?? 1}
                      onChange={(v) => setValues((s) => ({ ...s, [f.id]: v }))}
                    />
                  )}
                </Field>
              ))}
            </div>
          ) : (
            <p className="tiny dim">
              <b>{spec?.name}</b> needs no configuration — {spec?.blurb}
            </p>
          )}

          <hr className="divider" />

          <h4 className="section-title">Placement</h4>
          <div className="col" style={{ gap: 10 }}>
            <Field label="How many">
              <NumberInput value={quantity} min={1} max={10} onChange={setQuantity} />
            </Field>
            <Field label="Where">
              <Select value={placement} onChange={setPlacement} options={positions} />
            </Field>
            {placement === 'after' && (
              <Field label="Position" hint={`0 = before the first layer, ${layerCount} = at the end`}>
                <NumberInput
                  value={index}
                  min={0}
                  max={Math.max(0, layerCount)}
                  onChange={setIndex}
                />
              </Field>
            )}
          </div>
        </aside>
      </div>
    </Modal>
  );
}

function layerLabelSafe(type) {
  return LAYER_BY_TYPE[type]?.name || 'layer';
}
