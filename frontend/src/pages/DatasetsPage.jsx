import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import Icon from '../components/Icon';
import DatasetDialog from '../components/datasets/DatasetDialog';
import SampleEditor from '../components/datasets/SampleEditor';
import { Badge, Button, Card, EmptyState, SearchInput, Spinner } from '../components/ui';
import { useCatalog } from '../state/CatalogContext';
import { useSessionStore } from '../state/SessionContext';
import { useConfirm, useToast } from '../state/ToastContext';
import { fmtDate, fmtInt } from '../lib/format';

const TYPE_LABEL = {
  tabular: 'tabular',
  mnist: 'MNIST',
  fashion_mnist: 'Fashion-MNIST',
  image: 'image',
  custom: 'custom',
};

/**
 * DatasetsPage — the data library.
 *
 * Predefined datasets are read-only reference material; user datasets can be
 * edited sample by sample (table, pixel editor, scatter or raw JSON) and sent
 * straight to the studio.
 */
export default function DatasetsPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const confirm = useConfirm();
  const store = useSessionStore();
  const catalog = useCatalog();

  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [draft, setDraft] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dialog, setDialog] = useState({ open: false, dataset: null });

  const items = catalog.datasets || [];

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((d) =>
      [d.name, d.description, d.ds_type, TYPE_LABEL[d.ds_type]].join(' ').toLowerCase().includes(q),
    );
  }, [items, query]);

  const selected = items.find((d) => String(d.id) === String(selectedId)) || null;
  const readOnly = Boolean(selected?.is_predefined);
  const dirty = Boolean(detail && draft && JSON.stringify(draft) !== JSON.stringify(detail.data || []));

  const loadDetail = useCallback(
    async (id) => {
      if (!id) return;
      setLoading(true);
      try {
        const data = await api.getDataset(id);
        const ds = data.dataset || data;
        setDetail(ds);
        setDraft(Array.isArray(ds.data) ? ds.data : []);
      } catch (e) {
        toast.error(e.message);
      } finally {
        setLoading(false);
      }
    },
    [toast],
  );

  // Select the first dataset once the catalogue arrives.
  useEffect(() => {
    if (selectedId === null && items.length) setSelectedId(items[0].id);
  }, [items, selectedId]);

  useEffect(() => {
    if (selectedId) loadDetail(selectedId);
    else {
      setDetail(null);
      setDraft(null);
    }
  }, [selectedId, loadDetail]);

  async function save() {
    if (!selected || readOnly) return;
    setSaving(true);
    try {
      await api.updateDataset(selected.id, { data: draft });
      await loadDetail(selected.id);
      await catalog.refreshDatasets();
      toast.success(`Saved ${draft.length} samples to “${selected.name}”`);
    } catch (e) {
      toast.error(e.message);
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!selected || readOnly) return;
    const yes = await confirm({
      title: `Delete “${selected.name}”?`,
      message: 'This removes the dataset and its samples from your library. Models trained on it are unaffected.',
      confirmLabel: 'Delete dataset',
    });
    if (!yes) return;
    try {
      await api.deleteDataset(selected.id);
      await catalog.refreshDatasets();
      setSelectedId(null);
      toast.success('Dataset deleted');
    } catch (e) {
      toast.error(e.message);
    }
  }

  async function download() {
    if (!selected) return;
    setLoading(true);
    try {
      await api.downloadDataset(selected.id);
      await catalog.refreshDatasets();
      await loadDetail(selected.id);
      toast.success(`“${selected.name}” downloaded`);
    } catch (e) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }

  function trainOn() {
    if (!selected) return;
    if (!selected.downloaded || !(detail?.data?.length)) {
      toast.warn('Download this dataset before training on it.');
      return;
    }
    store.setConfig(
      {
        dsId: String(selected.id),
        funcKey: '',
        inputs: selected.num_inputs,
        outputs: selected.num_outputs || 1,
      },
      { markDirty: true },
    );
    store.syncIoDims({ inputs: selected.num_inputs, outputs: selected.num_outputs || 1 });
    store.pushHistory(`Switched to dataset “${selected.name}”`);
    navigate('/train');
    toast.info(`Studio is now pointed at “${selected.name}” — press Build.`);
  }

  function exportJson() {
    if (!draft) return;
    const blob = JSON.stringify({ name: selected?.name, data: draft }, null, 2);
    const url = URL.createObjectURL(new Blob([blob], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(selected?.name || 'dataset').toLowerCase().replace(/\W+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page datasets-page">
      <section className="datasets-page__list">
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <SearchInput value={query} onChange={setQuery} placeholder="Search datasets…" className="grow" />
          <Button variant="primary" icon="plus" onClick={() => setDialog({ open: true, dataset: null })}>
            New
          </Button>
        </div>

        <div className="ds-list">
          {filtered.map((d) => {
            const active = String(d.id) === String(selectedId);
            return (
              <button
                key={d.id}
                type="button"
                className={`ds-card ${active ? 'ds-card--active' : ''}`}
                onClick={() => setSelectedId(d.id)}
              >
                <span className="ds-card__icon">
                  <Icon name={d.width ? 'image' : 'database'} size={15} />
                </span>
                <span className="ds-card__body">
                  <span className="ds-card__name">{d.name}</span>
                  <span className="ds-card__meta mono">
                    {TYPE_LABEL[d.ds_type] || d.ds_type} · {fmtInt(d.num_inputs)} in
                    {d.num_outputs ? ` · ${fmtInt(d.num_outputs)} out` : ''} · {fmtInt(d.data_length)} samples
                  </span>
                </span>
                <span className="ds-card__flags">
                  {d.is_predefined ? <Badge tone="cyan">built-in</Badge> : <Badge tone="violet">yours</Badge>}
                  {!d.downloaded && <Badge tone="warn">not downloaded</Badge>}
                </span>
              </button>
            );
          })}
          {!filtered.length && (
            <EmptyState icon="search" title="No datasets match" message="Try a different search, or create a new dataset." />
          )}
        </div>

        <p className="tiny muted">
          <Icon name="info" size={12} /> Built-in datasets are shared reference data and stay read-only.
          Anything you create is yours to edit.
        </p>
      </section>

      <section className="datasets-page__detail">
        {!selected ? (
          <EmptyState
            icon="database"
            title="Pick a dataset"
            message="Choose one from the list to inspect its samples, or create your own."
          />
        ) : (
          <>
            <header className="ds-head">
              <div className="grow">
                <div className="row wrap" style={{ gap: 6 }}>
                  <h2>{selected.name}</h2>
                  <Badge tone={selected.is_predefined ? 'cyan' : 'violet'}>
                    {selected.is_predefined ? 'built-in' : 'your dataset'}
                  </Badge>
                  <Badge mono>{TYPE_LABEL[selected.ds_type] || selected.ds_type}</Badge>
                  {selected.width ? (
                    <Badge mono>
                      {selected.width}×{selected.height}×{selected.channels || 1}
                    </Badge>
                  ) : null}
                </div>
                {selected.description && <p className="ds-head__desc">{selected.description}</p>}
                <p className="tiny muted">
                  {fmtInt(selected.data_length)} samples ·{' '}
                  {selected.is_input_only ? 'inputs only' : `${fmtInt(selected.num_inputs)} inputs → ${fmtInt(selected.num_outputs)} outputs`}
                  {selected.updated_at ? ` · updated ${fmtDate(selected.updated_at)}` : ''}
                </p>
              </div>
              <div className="row wrap" style={{ gap: 6 }}>
                <Button variant="primary" icon="build" onClick={trainOn}>
                  Train on this
                </Button>
                {!selected.downloaded && (
                  <Button icon="download" loading={loading} onClick={download}>
                    Download
                  </Button>
                )}
                <Button variant="ghost" icon="download" onClick={exportJson} title="Export the samples as JSON">
                  Export
                </Button>
                {!readOnly && (
                  <>
                    <Button variant="ghost" icon="settings" onClick={() => setDialog({ open: true, dataset: selected })}>
                      Edit details
                    </Button>
                    <Button variant="danger" icon="trash" onClick={remove}>
                      Delete
                    </Button>
                  </>
                )}
              </div>
            </header>

            {loading ? (
              <div className="center" style={{ padding: 32 }}>
                <Spinner label="Loading samples…" />
              </div>
            ) : (
              <Card padded={false} headless>
                <SampleEditor dataset={detail || selected} readOnly={readOnly} data={draft || []} onChange={setDraft} />
              </Card>
            )}

            {!readOnly && (
              <footer className="ds-foot">
                <span className={`tiny ${dirty ? 'warn' : 'muted'}`}>
                  {dirty ? 'Unsaved changes in this dataset.' : 'All changes saved.'}
                </span>
                <div className="row" style={{ gap: 6 }}>
                  <Button variant="ghost" size="sm" disabled={!dirty} onClick={() => setDraft(detail?.data || [])}>
                    Discard
                  </Button>
                  <Button variant="primary" size="sm" icon="save" disabled={!dirty} loading={saving} onClick={save}>
                    Save samples
                  </Button>
                </div>
              </footer>
            )}
          </>
        )}
      </section>

      <DatasetDialog
        open={dialog.open}
        dataset={dialog.dataset}
        onClose={() => setDialog({ open: false, dataset: null })}
        onSaved={(id) => id && setSelectedId(id)}
      />
    </div>
  );
}
