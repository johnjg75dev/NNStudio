import { useEffect, useMemo, useState } from 'react';
import api from '../../api/client';
import Icon from '../Icon';
import { Badge, Button, Field, NumberInput, Popover, Segmented, Select, Switch, TextInput } from '../ui';
import { useCatalog } from '../../state/CatalogContext';
import { useSessionStore } from '../../state/SessionContext';
import { useConfirm, useToast } from '../../state/ToastContext';
import { fmtNum } from '../../lib/format';
import { useNavigate } from 'react-router-dom';

const STRATEGIES = [
  { value: 'linspace', label: 'Even grid (linspace)' },
  { value: 'random', label: 'Random sampling' },
  { value: 'custom', label: 'Fixed sample list' },
];

const blank = {
  name: '',
  description: '',
  language: 'python',
  code: '',
  num_inputs: 2,
  num_outputs: 1,
  input_labels: '',
  output_labels: '',
  is_classification: false,
  sample_strategy: 'linspace',
};

/** Editor for one custom training function: metadata, code, test run, dataset preview. */
export default function FunctionEditor({ func, templates, onSaved, onDeleted, onNew }) {
  const toast = useToast();
  const confirm = useConfirm();
  const catalog = useCatalog();
  const store = useSessionStore();
  const navigate = useNavigate();

  const [form, setForm] = useState(blank);
  const [busy, setBusy] = useState(null);
  const [testInput, setTestInput] = useState([0, 0]);
  const [testResult, setTestResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [samplesPerInput, setSamplesPerInput] = useState(5);

  useEffect(() => {
    if (!func) {
      setForm({ ...blank, code: templates?.templates?.python?.code || '' });
      setTestResult(null);
      setPreview(null);
      return;
    }
    setForm({
      name: func.name || '',
      description: func.description || '',
      language: func.language || 'python',
      code: func.code || '',
      num_inputs: func.num_inputs ?? 2,
      num_outputs: func.num_outputs ?? 1,
      input_labels: (func.input_labels || []).join(', '),
      output_labels: (func.output_labels || []).join(', '),
      is_classification: Boolean(func.is_classification),
      sample_strategy: func.sample_strategy || 'linspace',
    });
    setTestInput(new Array(func.num_inputs ?? 2).fill(0).map((_, i) => (i + 1) / (func.num_inputs ?? 2)));
    setTestResult(func.last_test_result || null);
    setPreview(null);
  }, [func, templates]);

  const patch = (p) => setForm((f) => ({ ...f, ...p }));
  const split = (s) =>
    String(s || '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

  const payload = useMemo(
    () => ({
      name: form.name.trim(),
      description: form.description.trim(),
      language: form.language,
      code: form.code,
      num_inputs: Number(form.num_inputs) || 1,
      num_outputs: Number(form.num_outputs) || 1,
      input_labels: split(form.input_labels),
      output_labels: split(form.output_labels),
      is_classification: form.is_classification,
      sample_strategy: form.sample_strategy,
    }),
    [form],
  );

  const examples = templates?.examples?.[form.language] || [];

  async function save() {
    if (!payload.name) return toast.warn('Name the function first.');
    if (!payload.code.trim()) return toast.warn('The function body cannot be empty.');
    setBusy('save');
    try {
      if (func?.id) {
        await api.updateFunction(func.id, payload);
        toast.success(`“${payload.name}” updated`);
      } else {
        await api.createFunction(payload);
        toast.success(`“${payload.name}” created`);
      }
      await catalog.refreshCustomFunctions();
      await catalog.refreshRegistry();
      onSaved?.();
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runTest() {
    if (!func?.id) {
      toast.warn('Save the function before testing it.');
      return;
    }
    setBusy('test');
    try {
      const res = await api.testFunction(func.id, testInput);
      setTestResult({ success: true, input: testInput, output: res.output, exec_time: res.exec_time });
      toast.success(`f([${testInput.map(fmtNum).join(', ')}]) → [${(res.output || []).map(fmtNum).join(', ')}]`);
    } catch (e) {
      setTestResult({ success: false, input: testInput, error: e.message });
      toast.error(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runPreview() {
    if (!func?.id) {
      toast.warn('Save the function before previewing its dataset.');
      return;
    }
    setBusy('preview');
    try {
      const res = await api.previewFunction(func.id, { samples_per_input: samplesPerInput, strategy: form.sample_strategy });
      setPreview(res);
    } catch (e) {
      toast.error(e.message);
      setPreview(null);
    } finally {
      setBusy(null);
    }
  }

  async function remove() {
    if (!func?.id) return;
    const yes = await confirm({
      title: `Delete “${func.name}”?`,
      message: 'Networks already trained on it keep their weights, but the task disappears from the picker.',
      confirmLabel: 'Delete function',
    });
    if (!yes) return;
    try {
      await api.deleteFunction(func.id);
      await catalog.refreshCustomFunctions();
      await catalog.refreshRegistry();
      toast.success('Function deleted');
      onDeleted?.();
    } catch (e) {
      toast.error(e.message);
    }
  }

  function trainOn() {
    if (!func?.id) return;
    store.setConfig(
      {
        funcKey: `custom_${func.id}`,
        dsId: '',
        inputs: func.num_inputs,
        outputs: func.num_outputs,
      },
      { markDirty: true },
    );
    store.syncIoDims({ inputs: func.num_inputs, outputs: func.num_outputs });
    store.pushHistory(`Switched to custom function “${func.name}”`);
    navigate('/train');
    toast.info('Studio is pointed at your function — press Build.');
  }

  return (
    <div className="fn-editor">
      <div className="fn-editor__grid">
        <Field label="Name">
          <TextInput value={form.name} onChange={(v) => patch({ name: v })} placeholder="Spiral classifier" />
        </Field>
        <Field label="Language">
          <Segmented
            value={form.language}
            onChange={(v) =>
              patch({
                language: v,
                code: form.code || templates?.templates?.[v]?.code || '',
              })
            }
            options={[
              { value: 'python', label: 'Python', icon: 'code' },
              { value: 'javascript', label: 'JavaScript', icon: 'code' },
            ]}
          />
        </Field>
      </div>

      <Field label="Description" hint="Shown in the task picker.">
        <TextInput value={form.description} onChange={(v) => patch({ description: v })} placeholder="Two interleaved spirals → class" />
      </Field>

      <div className="fn-editor__dims">
        <Field label="Inputs" style={{ width: 96 }}>
          <NumberInput
            value={form.num_inputs}
            min={1}
            max={32}
            onChange={(v) => {
              patch({ num_inputs: v });
              setTestInput(new Array(Number(v) || 1).fill(0.5));
            }}
          />
        </Field>
        <Field label="Outputs" style={{ width: 96 }}>
          <NumberInput value={form.num_outputs} min={1} max={32} onChange={(v) => patch({ num_outputs: v })} />
        </Field>
        <Field label="Input labels" className="grow" hint="Comma separated">
          <TextInput value={form.input_labels} onChange={(v) => patch({ input_labels: v })} placeholder="x, y" />
        </Field>
        <Field label="Output labels" className="grow" hint="Comma separated">
          <TextInput value={form.output_labels} onChange={(v) => patch({ output_labels: v })} placeholder="class" />
        </Field>
      </div>

      <div className="field">
        <div className="row row--between">
          <span className="field__label">
            Function body <span className="mono tiny muted">f(x) → [{form.num_outputs} value{(form.num_outputs || 1) > 1 ? 's' : ''}]</span>
          </span>
          <div className="row" style={{ gap: 6 }}>
            {examples.length > 0 && (
              <Popover
                width={280}
                trigger={(open) => (
                  <Button size="xs" variant="ghost" icon="sparkles">
                    Examples <Icon name={open ? 'chevronUp' : 'chevronDown'} size={12} />
                  </Button>
                )}
              >
                {(close) => (
                  <div className="col" style={{ gap: 4 }}>
                    <button type="button" className="menu-item" onClick={() => { patch({ code: templates?.templates?.[form.language]?.code || '' }); close(); }}>
                      <span className="tiny">Starter template</span>
                    </button>
                    {examples.map((ex) => (
                      <button key={ex.name} type="button" className="menu-item" onClick={() => { patch({ code: ex.code }); close(); }}>
                        <span className="tiny">{ex.name}</span>
                        <span className="mono xs muted truncate">{ex.code.split('\n')[1]?.trim() || ''}</span>
                      </button>
                    ))}
                  </div>
                )}
              </Popover>
            )}
            <Badge mono>{form.code.split('\n').length} lines</Badge>
          </div>
        </div>
        <textarea
          className="textarea textarea--code mono"
          rows={12}
          spellCheck={false}
          value={form.code}
          placeholder={form.language === 'python' ? 'def f(x):\n    return [x[0] * x[1]]' : 'function f(x) {\n    return [x[0] * x[1]];\n}'}
          onChange={(e) => patch({ code: e.target.value })}
        />
        <span className="field__hint">
          {form.language === 'python'
            ? 'Define f(x). NumPy is available; return a list of numbers.'
            : 'Define f(x) in JavaScript; return an array of numbers.'}
        </span>
      </div>

      <div className="row wrap" style={{ gap: 14, alignItems: 'center' }}>
        <Switch
          checked={form.is_classification}
          onChange={(v) => patch({ is_classification: v })}
          label="Classification task"
          tip="Marks the task as classification so accuracy is reported instead of just loss."
        />
        <Field label="Sample strategy" className="grow" style={{ maxWidth: 260 }}>
          <Select value={form.sample_strategy} onChange={(v) => patch({ sample_strategy: v })} options={STRATEGIES} />
        </Field>
        <Field label="Samples per input" style={{ width: 140 }}>
          <NumberInput value={samplesPerInput} min={2} max={20} onChange={setSamplesPerInput} />
        </Field>
      </div>

      <div className="row wrap fn-editor__actions">
        <Button variant="primary" icon="save" loading={busy === 'save'} onClick={save}>
          {func?.id ? 'Save changes' : 'Create function'}
        </Button>
        <Button icon="bolt" loading={busy === 'test'} onClick={runTest} disabled={!func?.id}>
          Test run
        </Button>
        <Button icon="database" loading={busy === 'preview'} onClick={runPreview} disabled={!func?.id}>
          Preview dataset
        </Button>
        <Button icon="build" onClick={trainOn} disabled={!func?.id}>
          Train on this
        </Button>
        {func?.id && (
          <Button variant="danger" icon="trash" onClick={remove}>
            Delete
          </Button>
        )}
        {!func?.id && (
          <Button variant="ghost" icon="close" onClick={onNew}>
            Discard draft
          </Button>
        )}
      </div>

      {func?.id === undefined && (
        <div className="hint-row">
          <Icon name="info" size={14} />
          <span>New function — save it first, then test and preview. Code is validated on the server.</span>
        </div>
      )}

      <TestBench
        n={Number(form.num_inputs) || 1}
        value={testInput}
        onChange={setTestInput}
        result={testResult}
        onRun={runTest}
        busy={busy === 'test'}
        disabled={!func?.id}
      />

      {preview && (
        <section className="fn-preview">
          <header className="sec-head">
            <h3>Dataset preview</h3>
            <Badge mono>{preview.total_samples} samples generated</Badge>
          </header>
          <div className="table-scroll table-scroll--short">
            <table className="table table--io">
              <thead>
                <tr>
                  <th>#</th>
                  <th>x</th>
                  <th>y</th>
                </tr>
              </thead>
              <tbody>
                {(preview.preview || []).map((s, i) => (
                  <tr key={i}>
                    <td className="mono muted">{i}</td>
                    <td className="mono tiny">{(s.x || []).map(fmtNum).join(' · ')}</td>
                    <td className="mono tiny">{(s.y || []).map(fmtNum).join(' · ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny muted">
            Showing the first {(preview.preview || []).length} of {preview.total_samples} samples the generator
            would produce with the “{form.sample_strategy}” strategy.
          </p>
        </section>
      )}
    </div>
  );
}

/** Small input vector + result readout for one-off function runs. */
function TestBench({ n, value, onChange, result, onRun, busy, disabled }) {
  const inputs = Array.from({ length: n }, (_, i) => Number(value[i]) || 0);
  return (
    <section className="test-bench">
      <header className="sec-head">
        <h3>Test bench</h3>
        {result?.exec_time !== undefined && result?.exec_time !== null && (
          <Badge mono>{(result.exec_time * 1000).toFixed(2)} ms</Badge>
        )}
      </header>
      <div className="row wrap" style={{ gap: 6 }}>
        {inputs.map((v, i) => (
          <label key={i} className="test-bench__cell">
            <span className="mono xs muted">x{i}</span>
            <NumberInput
              className="input--xs mono"
              value={v}
              step={0.1}
              onChange={(nv) => {
                const next = [...inputs];
                next[i] = nv === '' ? 0 : nv;
                onChange(next);
              }}
            />
          </label>
        ))}
        <Button size="sm" variant="primary" icon="play" loading={busy} disabled={disabled} onClick={onRun}>
          Run f(x)
        </Button>
      </div>
      {result && (
        <div className={`banner ${result.success === false ? 'banner--neg' : 'banner--pos'}`}>
          <Icon name={result.success === false ? 'alert' : 'check'} size={14} />
          {result.success === false ? (
            <span className="mono tiny">{result.error}</span>
          ) : (
            <span className="mono tiny">
              → [{(result.output || []).map(fmtNum).join(', ')}]
            </span>
          )}
        </div>
      )}
      {disabled && <p className="tiny muted">Save the function to enable test runs.</p>}
    </section>
  );
}
