import { useEffect, useMemo, useState } from 'react';
import api from '../api/client';
import Icon from '../components/Icon';
import FunctionEditor from '../components/functions/FunctionEditor';
import { Badge, Button, Card, EmptyState, HtmlText, SearchInput, Spinner } from '../components/ui';
import { useCatalog } from '../state/CatalogContext';
import { useToast } from '../state/ToastContext';
import { fmtDate, fmtInt } from '../lib/format';

/**
 * FunctionsPage — write your own training tasks.
 *
 * A custom function defines the target output for any input vector; the studio
 * generates a dataset from it and trains against it like any built-in task.
 */
export default function FunctionsPage() {
  const catalog = useCatalog();
  const toast = useToast();

  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [draft, setDraft] = useState(false);
  const [templates, setTemplates] = useState(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');

  const custom = catalog.customFunctions || [];
  const builtin = catalog.functions || [];

  useEffect(() => {
    api
      .functionTemplates()
      .then(setTemplates)
      .catch(() => setTemplates(null));
  }, []);

  useEffect(() => {
    if (draft || !selectedId) {
      setDetail(null);
      return;
    }
    let alive = true;
    setLoading(true);
    api
      .getFunction(selectedId)
      .then((res) => {
        if (alive) setDetail(res.function || res);
      })
      .catch((e) => alive && toast.error(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [selectedId, draft, toast]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return custom;
    return custom.filter((f) => [f.name, f.description, f.language].join(' ').toLowerCase().includes(q));
  }, [custom, query]);

  return (
    <div className="page functions-page">
      <aside className="functions-page__list">
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <SearchInput value={query} onChange={setQuery} placeholder="Search functions…" className="grow" />
          <Button
            variant="primary"
            icon="plus"
            onClick={() => {
              setDraft(true);
              setSelectedId(null);
            }}
          >
            New
          </Button>
        </div>

        <div className="ds-list">
          {filtered.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`ds-card ${!draft && String(f.id) === String(selectedId) ? 'ds-card--active' : ''}`}
              onClick={() => {
                setDraft(false);
                setSelectedId(f.id);
              }}
            >
              <span className="ds-card__icon">
                <Icon name="code" size={15} />
              </span>
              <span className="ds-card__body">
                <span className="ds-card__name">{f.name}</span>
                <span className="ds-card__meta mono">
                  {f.language} · {fmtInt(f.num_inputs)} in → {fmtInt(f.num_outputs)} out ·{' '}
                  {fmtDate(f.updated_at || f.created_at)}
                </span>
              </span>
              <span className="ds-card__flags">
                {f.is_valid ? <Badge tone="pos">valid</Badge> : <Badge tone="neg">errors</Badge>}
              </span>
            </button>
          ))}
          {!filtered.length && (
            <EmptyState
              icon="code"
              title={custom.length ? 'No functions match' : 'No custom functions yet'}
              message={
                custom.length
                  ? 'Try another search term.'
                  : 'Write one and it becomes a trainable task — the studio generates its dataset for you.'
              }
              action={
                <Button
                  variant="primary"
                  icon="plus"
                  onClick={() => {
                    setDraft(true);
                    setSelectedId(null);
                  }}
                >
                  New function
                </Button>
              }
            />
          )}
        </div>

        <Card title="Built-in tasks" subtitle="Reference — these ship with the app" className="mt-12">
          <ul className="ref-list">
            {builtin.map((f) => (
              <li key={f.key} className="ref-list__item">
                <div className="row" style={{ gap: 6 }}>
                  <span className="tiny strong">{f.label}</span>
                  <Badge mono>
                    {f.inputs}→{f.outputs}
                  </Badge>
                </div>
                {f.description && <HtmlText html={f.description} className="tiny muted ref-list__desc" />}
              </li>
            ))}
          </ul>
        </Card>
      </aside>

      <section className="functions-page__editor">
        {loading ? (
          <div className="center" style={{ padding: 40 }}>
            <Spinner label="Loading function…" />
          </div>
        ) : (
          <Card
            title={draft ? 'New custom function' : detail?.name || 'Custom function'}
            subtitle={
              draft
                ? 'Define the target output for any input vector'
                : detail?.description || 'Edit the definition, then test it before training'
            }
            actions={
              detail ? (
                <span className="row" style={{ gap: 6 }}>
                  <Badge tone={detail.language === 'python' ? 'accent' : 'warn'}>{detail.language}</Badge>
                  <Badge mono>custom_{detail.id}</Badge>
                </span>
              ) : null
            }
          >
            <FunctionEditor
              func={draft ? undefined : detail}
              templates={templates}
              onNew={() => {
                setDraft(false);
                setSelectedId(custom[0]?.id ?? null);
              }}
              onSaved={() => {
                setDraft(false);
                catalog.refreshCustomFunctions();
              }}
              onDeleted={() => {
                setDraft(false);
                setSelectedId(null);
                setDetail(null);
              }}
            />
          </Card>
        )}
      </section>
    </div>
  );
}
