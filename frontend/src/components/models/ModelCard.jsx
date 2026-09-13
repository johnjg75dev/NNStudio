import { useState } from 'react';
import api from '../../api/client';
import Icon from '../Icon';
import { Badge, Button, Metric, Popover } from '../ui';
import { useToast } from '../../state/ToastContext';
import { fmtDate, fmtInt, fmtLoss, fmtPct } from '../../lib/format';

const FORMAT_LABEL = {
  json: 'JSON — full serialisation',
  safetensors: 'SafeTensors — Hugging Face',
  gguf: 'GGUF — quantised inference',
  onnx: 'ONNX — cross-platform',
  zip: 'ZIP — model + weights + metadata',
};

/** One saved model in the library, with load / export / delete actions. */
export default function ModelCard({ model, active, onLoad, onDeleted }) {
  const toast = useToast();
  const [busy, setBusy] = useState(null);
  const [detail, setDetail] = useState(null);

  async function inspect() {
    setDetail(null);
    if (detail) return;
    setBusy('inspect');
    try {
      const res = await api.getModel(model.id);
      const m = res.model || res;
      const data = m.model_data || {};
      setDetail({
        topology: data.topology || null,
        layers: (data.layers || []).map((l) => ({
          type: l.type || l.class || 'layer',
          activation: l.activation,
          n_in: l.n_in,
          n_out: l.n_out,
        })),
        epoch: data.epoch,
        params: data.param_count ?? null,
        lossHistory: (data.loss_history || []).slice(-40),
      });
    } catch (e) {
      setDetail({ error: e.message });
    } finally {
      setBusy(null);
    }
  }

  return (
    <article className={`model-card ${active ? 'model-card--active' : ''}`}>
      <header className="model-card__head">
        <div className="grow">
          <h3>{model.name}</h3>
          {model.description && <p className="tiny muted truncate">{model.description}</p>}
        </div>
        <div className="row wrap" style={{ gap: 4 }}>
          <Badge tone="accent">{model.architecture_name || 'mlp'}</Badge>
          <Badge mono>{model.function_name || '—'}</Badge>
        </div>
      </header>

      <div className="metrics metrics--sm">
        <Metric label="Epochs" value={fmtInt(model.epochs_trained)} />
        <Metric label="Loss" value={fmtLoss(model.final_loss)} />
        <Metric
          label="Accuracy"
          value={model.final_accuracy === null || model.final_accuracy === undefined ? '—' : fmtPct(model.final_accuracy)}
          tone={model.final_accuracy >= 0.999 ? 'pos' : ''}
        />
      </div>

      <footer className="model-card__foot">
        <span className="tiny muted">{fmtDate(model.created_at)}</span>
        <div className="row" style={{ gap: 6, marginLeft: 'auto' }}>
          <Button size="sm" variant="ghost" icon="eye" onClick={inspect} loading={busy === 'inspect'}>
            {detail ? 'Hide' : 'Inspect'}
          </Button>
          <Popover
            width={260}
            trigger={(open) => (
              <Button size="sm" variant="ghost" icon="download" title="Export in another format">
                Export <Icon name={open ? 'chevronUp' : 'chevronDown'} size={12} />
              </Button>
            )}
          >
            {(close) => (
              <div className="col" style={{ gap: 4 }}>
                {Object.entries(FORMAT_LABEL).map(([fmt, label]) => (
                  <button
                    key={fmt}
                    type="button"
                    className="menu-item"
                    onClick={async () => {
                      close();
                      setBusy('export');
                      try {
                        const res = await api.exportModel(model.id, fmt);
                        const url = res.download_url || api.modelDownloadUrl(model.id, fmt);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = '';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                      } catch (e) {
                        toast.error(e.message);
                      } finally {
                        setBusy(null);
                      }
                    }}
                  >
                    <span className="mono tiny">{fmt}</span>
                    <span className="tiny muted">{label.split('—')[1]?.trim() || label}</span>
                  </button>
                ))}
              </div>
            )}
          </Popover>
          <Button size="sm" variant="primary" icon="upload" loading={busy === 'load'} onClick={() => onLoad(model)}>
            Load
          </Button>
          <Button size="sm" variant="danger" iconOnly title="Delete model" onClick={() => onDeleted(model)}>
            <Icon name="trash" />
          </Button>
        </div>
      </footer>

      {detail && (
        <div className="model-card__detail">
          {detail.error ? (
            <p className="tiny neg">{detail.error}</p>
          ) : (
            <>
              <div className="row wrap" style={{ gap: 6 }}>
                {detail.topology && <Badge mono>[{detail.topology.join(' → ')}]</Badge>}
                {detail.params !== null && <Badge mono>{fmtInt(detail.params)} params</Badge>}
                <Badge mono>{detail.layers.length} layers</Badge>
              </div>
              <ul className="layer-list">
                {detail.layers.map((l, i) => (
                  <li key={i} className="mono tiny">
                    <span className="badge">{l.type}</span>
                    {l.n_in ? `${l.n_in}→${l.n_out || '?'}` : ''}
                    {typeof l.activation === 'string' ? l.activation : Array.isArray(l.activation) ? l.activation[0] : ''}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </article>
  );
}
