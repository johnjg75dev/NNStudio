import { useMemo, useState } from 'react';
import api from '../../api/client';
import { Badge, Button, EmptyState, Modal, SearchInput } from '../ui';
import Icon from '../Icon';
import { useCatalog } from '../../state/CatalogContext';
import { useToast, useConfirm } from '../../state/ToastContext';
import { LAYER_BY_TYPE, layerLabel } from '../../lib/layers';

/** Preset gallery: one click loads a full configuration. */
export default function PresetGallery({ open, onClose, onApply, onSaveCurrent }) {
  const { presets, refreshRegistry } = useCatalog();
  const toast = useToast();
  const confirm = useConfirm();
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return presets;
    return presets.filter(
      (p) =>
        p.label?.toLowerCase().includes(q) ||
        p.description?.toLowerCase().includes(q) ||
        p.func_key?.toLowerCase().includes(q) ||
        p.arch_key?.toLowerCase().includes(q),
    );
  }, [presets, query]);

  async function remove(preset) {
    const yes = await confirm({
      title: `Delete “${preset.label}”?`,
      message: 'This removes the preset from your account. It cannot be undone.',
      confirmLabel: 'Delete preset',
    });
    if (!yes) return;
    try {
      await api.deletePreset(preset.id);
      await refreshRegistry();
      toast.success(`Preset “${preset.label}” deleted`);
    } catch (e) {
      toast.error(e.message);
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Presets"
      subtitle="A preset fills in the task, layer stack and hyperparameters in one click."
      icon="sparkles"
      size="wide"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" icon="save" onClick={onSaveCurrent}>
            Save current setup
          </Button>
        </>
      }
    >
      <div className="picker">
        <SearchInput value={query} onChange={setQuery} placeholder="Search presets…" />
        {!filtered.length && (
          <EmptyState icon="sparkles" title="No presets match">
            Save your current setup to build your own library.
          </EmptyState>
        )}
        <div className="preset-grid">
          {filtered.map((p) => (
            <article key={p.key || p.id || p.label} className="preset-card card card--hover">
              <header>
                <h4>{p.label}</h4>
                <div className="row" style={{ gap: 4 }}>
                  <Badge tone="accent">{p.arch_key}</Badge>
                  <Badge>{p.func_key}</Badge>
                  {p.custom && (
                    <button
                      className="tool-btn"
                      title="Delete preset"
                      onClick={(e) => {
                        e.stopPropagation();
                        remove(p);
                      }}
                    >
                      <Icon name="trash" size={13} />
                    </button>
                  )}
                </div>
              </header>
              <p className="tiny dim">{p.description || 'No description.'}</p>
              <div className="preset-card__layers">
                {(p.layers || []).map((l, i) => (
                  <span key={i} className="badge badge--mono">
                    <Icon name={LAYER_BY_TYPE[l.type]?.icon || 'layers'} size={10} />
                    {layerLabel(l.type)}
                    {l.neurons ? ` ×${l.neurons}` : ''}
                  </span>
                ))}
                {!(p.layers || []).length && <span className="tiny muted">no hidden layers</span>}
              </div>
              <dl className="preset-card__meta">
                <div>
                  <dt>optimizer</dt>
                  <dd>{p.optimizer}</dd>
                </div>
                <div>
                  <dt>loss</dt>
                  <dd>{p.loss}</dd>
                </div>
                <div>
                  <dt>lr</dt>
                  <dd>{p.lr}</dd>
                </div>
                <div>
                  <dt>decay</dt>
                  <dd>{p.weight_decay ?? 0}</dd>
                </div>
              </dl>
              <Button
                variant="primary"
                size="sm"
                block
                icon="bolt"
                onClick={() => {
                  onApply(p);
                  onClose();
                }}
              >
                Load preset
              </Button>
            </article>
          ))}
        </div>
      </div>
    </Modal>
  );
}
