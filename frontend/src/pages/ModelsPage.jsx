import { useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import Icon from '../components/Icon';
import ModelCard from '../components/models/ModelCard';
import { Badge, Button, Card, EmptyState, Field, Metric, SearchInput, Segmented, TextInput } from '../components/ui';
import { useCatalog } from '../state/CatalogContext';
import { useSession, useSessionStore } from '../state/SessionContext';
import { useConfirm, useToast } from '../state/ToastContext';
import { download, fmtCompact, fmtInt, fmtLoss, fmtPct, readFileAsText } from '../lib/format';

const SORTS = [
  { value: 'recent', label: 'Newest' },
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'loss', label: 'Loss' },
  { value: 'name', label: 'Name' },
];

/**
 * ModelsPage — save the live session, browse the library, load a model back
 * into the studio, and export in interchange formats.
 */
export default function ModelsPage() {
  const store = useSessionStore();
  const catalog = useCatalog();
  const toast = useToast();
  const confirm = useConfirm();
  const fileRef = useRef(null);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('recent');

  const snapshot = useSession((s) => s.snapshot);
  const metrics = useSession((s) => s.metrics);
  const stepsRun = useSession((s) => s.stepsRun);
  const built = Boolean(snapshot?.built);

  const models = catalog.models || [];

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = q
      ? models.filter((m) => [m.name, m.description, m.architecture_name, m.function_name].join(' ').toLowerCase().includes(q))
      : [...models];
    const by = {
      recent: (a, b) => new Date(b.created_at) - new Date(a.created_at),
      accuracy: (a, b) => (b.final_accuracy ?? -1) - (a.final_accuracy ?? -1),
      loss: (a, b) => (a.final_loss ?? Infinity) - (b.final_loss ?? Infinity),
      name: (a, b) => String(a.name).localeCompare(String(b.name)),
    }[sort];
    return list.sort(by);
  }, [models, query, sort]);

  async function save() {
    if (!built) {
      toast.warn('Build and train a network before saving it.');
      return;
    }
    const label = name.trim() || `${snapshot.func?.label || snapshot.func_key} · epoch ${metrics.epoch}`;
    setBusy(true);
    try {
      await api.saveModel({ name: label, description: description.trim() });
      await catalog.refreshModels();
      toast.success(`Saved “${label}” to your library`);
      setName('');
      setDescription('');
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function exportFile() {
    setBusy(true);
    try {
      const data = await store.exportModel();
      if (data) {
        download(
          `${(snapshot?.func?.label || 'model').toLowerCase().replace(/\W+/g, '-')}-epoch-${metrics.epoch}.json`,
          JSON.stringify(data, null, 2),
        );
        toast.success('Model exported as JSON');
      }
    } finally {
      setBusy(false);
    }
  }

  async function importFile(file) {
    if (!file) return;
    setBusy(true);
    try {
      const text = await readFileAsText(file);
      const data = JSON.parse(text);
      await store.importModel(data);
      toast.success(`Imported “${file.name}” into the session`);
    } catch (e) {
      toast.error(`Could not import that file: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function load(model) {
    setBusy(true);
    try {
      await store.loadLibraryModel(model.id);
      toast.success(`“${model.name}” is live in the studio`);
    } finally {
      setBusy(false);
    }
  }

  async function remove(model) {
    const yes = await confirm({
      title: `Delete “${model.name}”?`,
      message: 'The saved weights are removed from your library. This cannot be undone.',
      confirmLabel: 'Delete model',
    });
    if (!yes) return;
    try {
      await api.deleteModel(model.id);
      await catalog.refreshModels();
      toast.success('Model deleted');
    } catch (e) {
      toast.error(e.message);
    }
  }

  return (
    <div className="page models-page">
      <section className="models-page__session">
        <Card
          title="Current session"
          subtitle="Whatever is built in the studio right now"
          actions={
            built ? (
              <Badge tone="pos" mono>
                [{(snapshot.topology || []).join('→')}]
              </Badge>
            ) : (
              <Badge tone="warn">nothing built</Badge>
            )
          }
        >
          {built ? (
            <>
              <div className="metrics">
                <Metric label="Epoch" value={fmtInt(metrics.epoch)} tone="accent" />
                <Metric label="Loss" value={fmtLoss(metrics.loss)} />
                <Metric
                  label="Accuracy"
                  value={metrics.accuracy === null || metrics.accuracy === undefined ? '—' : fmtPct(metrics.accuracy)}
                />
                <Metric label="Parameters" value={fmtCompact(metrics.params)} />
                <Metric label="Steps run" value={fmtCompact(stepsRun)} />
              </div>

              <div className="col" style={{ gap: 10, marginTop: 14 }}>
                <Field label="Model name" hint="Leave blank to name it after the task and epoch.">
                  <TextInput value={name} onChange={setName} placeholder="XOR — 4 hidden, 200 epochs" />
                </Field>
                <Field label="Description">
                  <TextInput
                    value={description}
                    onChange={setDescription}
                    placeholder="What should future-you remember about this run?"
                  />
                </Field>
                <div className="row wrap" style={{ gap: 6 }}>
                  <Button variant="primary" icon="save" loading={busy} onClick={save}>
                    Save to library
                  </Button>
                  <Button icon="download" loading={busy} onClick={exportFile}>
                    Export JSON file
                  </Button>
                  <Button icon="upload" onClick={() => fileRef.current?.click()}>
                    Import from file
                  </Button>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="application/json,.json"
                    hidden
                    onChange={(e) => {
                      importFile(e.target.files?.[0]);
                      e.target.value = '';
                    }}
                  />
                  <Button as={Link} to="/train" variant="ghost" iconRight="arrowRight">
                    Back to the studio
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <EmptyState
              icon="network"
              title="No network in the session"
              message="Build a network on the Train page and it will appear here, ready to save, export or compare."
              action={
                <Button as={Link} to="/train" variant="primary" icon="build">
                  Open the studio
                </Button>
              }
            />
          )}
        </Card>
      </section>

      <section className="models-page__library">
        <div className="row wrap" style={{ gap: 8, marginBottom: 12 }}>
          <h2 className="section-heading">
            Library <Badge mono>{models.length}</Badge>
          </h2>
          <SearchInput value={query} onChange={setQuery} placeholder="Search models…" className="grow" />
          <Segmented value={sort} onChange={setSort} options={SORTS} />
        </div>

        {!visible.length ? (
          <EmptyState
            icon="save"
            title={models.length ? 'No models match that search' : 'Your library is empty'}
            message={
              models.length
                ? 'Try another term, or clear the search to see everything.'
                : 'Train something in the studio, then save it here. Saved models keep their weights, topology and history — and can be exported to JSON, SafeTensors, GGUF, ONNX or ZIP.'
            }
            action={
              models.length ? (
                <Button variant="ghost" icon="close" onClick={() => setQuery('')}>
                  Clear search
                </Button>
              ) : (
                <Button as={Link} to="/train" variant="primary" icon="build">
                  Start training
                </Button>
              )
            }
          />
        ) : (
          <div className="model-grid">
            {visible.map((m) => (
              <ModelCard
                key={m.id}
                model={m}
                active={built && snapshot?.func_key === m.function_name && metrics.epoch === m.epochs_trained}
                onLoad={load}
                onDeleted={remove}
              />
            ))}
          </div>
        )}

        <p className="tiny muted">
          <Icon name="info" size={12} /> Loading a model replaces the studio session — including its weights —
          so you can keep training from that point.
        </p>
      </section>
    </div>
  );
}
