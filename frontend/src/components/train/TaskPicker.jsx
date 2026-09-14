import { useMemo, useState } from 'react';
import { Badge, EmptyState, Modal, SearchInput } from '../ui';
import Icon from '../Icon';
import { useCatalog } from '../../state/CatalogContext';

/**
 * TaskPicker — one place to choose what the network should learn:
 * built-in training tasks, user datasets, or custom functions.
 */
export default function TaskPicker({ open, onClose, current, onSelect }) {
  const { functions, datasets, customFunctions } = useCatalog();
  const [query, setQuery] = useState('');

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (s) => !q || String(s || '').toLowerCase().includes(q);

    const byCategory = {};
    functions
      .filter((f) => match(f.label) || match(f.key) || match(f.description))
      .forEach((f) => {
        const cat = f.category === 'functions' ? f.key === 'xor' ? 'Logic gates' : 'Built-in tasks' : f.category;
        (byCategory[cat] ||= []).push(f);
      });

    const taskGroups = Object.entries(byCategory).map(([name, items]) => ({
      id: `fn-${name}`,
      title: name,
      items: items.map((f) => ({
        id: `fn:${f.key}`,
        kind: 'function',
        title: f.label,
        subtitle: `${f.inputs} in → ${f.outputs} out`,
        description: f.description,
        value: { funcKey: f.key, dsId: '', inputs: f.inputs, outputs: f.outputs, meta: f },
        active: current.funcKey === f.key && !current.dsId,
        badge: f.is_classification ? 'classification' : 'regression',
      })),
    }));

    const dsItems = datasets
      .filter((d) => match(d.name) || match(d.description))
      .map((d) => ({
        id: `ds:${d.id}`,
        kind: 'dataset',
        title: d.name,
        subtitle: `${d.ds_type} · ${d.num_inputs} in → ${d.num_outputs || 1} out · ${d.data_length ?? 0} rows`,
        description: d.description,
        value: {
          dsId: String(d.id),
          funcKey: '',
          inputs: d.num_inputs,
          outputs: d.num_outputs || 1,
          meta: d,
        },
        active: String(current.dsId) === String(d.id),
        badge: d.is_predefined ? 'predefined' : d.is_input_only ? 'input only' : 'yours',
        disabled: d.is_predefined && !d.downloaded,
        disabledReason: 'Download this predefined dataset first.',
      }));

    const cfItems = customFunctions
      .filter((f) => f.is_valid && match(f.name))
      .map((f) => ({
        id: `cf:${f.id}`,
        kind: 'function',
        title: f.name,
        subtitle: `${f.language} · ${f.num_inputs} in → ${f.num_outputs} out`,
        description: f.description,
        value: { funcKey: f.name, dsId: '', inputs: f.num_inputs, outputs: f.num_outputs, meta: f },
        active: current.funcKey === f.name && !current.dsId,
        badge: 'custom',
      }));

    const out = [];
    if (dsItems.length) out.push({ id: 'datasets', title: 'Your datasets', items: dsItems });
    if (cfItems.length) out.push({ id: 'custom', title: 'Custom functions', items: cfItems });
    out.push(...taskGroups);
    return out;
  }, [functions, datasets, customFunctions, query, current]);

  const total = groups.reduce((n, g) => n + g.items.length, 0);

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Choose a training task"
      subtitle="A task defines the inputs, the expected outputs and the data the network learns from."
      icon="target"
      size="wide"
    >
      <div className="picker">
        <SearchInput
          value={query}
          onChange={setQuery}
          placeholder="Search tasks, datasets and functions…"
          className="picker__search"
        />

        {total === 0 && (
          <EmptyState icon="search" title="Nothing matches that">
            Try a different search, or create a dataset / custom function first.
          </EmptyState>
        )}

        {groups.map((group) => (
          <section key={group.id} className="picker__group">
            <h4 className="section-title">{group.title}</h4>
            <div className="picker__grid">
              {group.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="pick"
                  data-active={item.active ? 'true' : 'false'}
                  disabled={item.disabled}
                  title={item.disabled ? item.disabledReason : undefined}
                  onClick={() => {
                    onSelect(item.value);
                    onClose();
                  }}
                >
                  <span className="pick__icon">
                    <Icon name={item.kind === 'dataset' ? 'database' : 'cpu'} size={16} />
                  </span>
                  <span className="pick__body">
                    <span className="pick__title">
                      {item.title}
                      {item.active && <Icon name="check" size={13} className="pos" />}
                    </span>
                    <span className="pick__sub mono">{item.subtitle}</span>
                    {item.description && (
                      <span
                        className="pick__desc"
                        dangerouslySetInnerHTML={{ __html: stripTags(item.description) }}
                      />
                    )}
                  </span>
                  {item.badge && <Badge tone={item.badge === 'custom' ? 'violet' : ''}>{item.badge}</Badge>}
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    </Modal>
  );
}

/** Registry descriptions contain HTML; keep the text, drop the markup. */
function stripTags(html) {
  return String(html || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
