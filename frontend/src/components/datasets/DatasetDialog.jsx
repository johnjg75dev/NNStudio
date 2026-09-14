import { useEffect, useMemo, useState } from 'react';
import api from '../../api/client';
import { Badge, Button, Field, Modal, NumberInput, Select, Switch, TextInput } from '../ui';
import { useCatalog } from '../../state/CatalogContext';
import { useToast } from '../../state/ToastContext';

const TYPES = [
  { value: 'tabular', label: 'Tabular — rows of numbers' },
  { value: 'image', label: 'Image — pixel grid input' },
  { value: 'custom', label: 'Custom — free-form samples' },
];

const empty = {
  name: '',
  description: '',
  ds_type: 'tabular',
  num_inputs: 2,
  num_outputs: 1,
  input_labels: '',
  output_labels: '',
  width: null,
  height: null,
  channels: 1,
  is_input_only: false,
  sampleCount: 8,
};

/** Create or edit a dataset's metadata (and seed its first samples). */
export default function DatasetDialog({ open, onClose, dataset = null, onSaved }) {
  const { refreshDatasets } = useCatalog();
  const toast = useToast();
  const editing = Boolean(dataset?.id);
  const [form, setForm] = useState(empty);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setForm(
      dataset
        ? {
            name: dataset.name || '',
            description: dataset.description || '',
            ds_type: dataset.ds_type || 'tabular',
            num_inputs: dataset.num_inputs ?? 2,
            num_outputs: dataset.num_outputs ?? 1,
            input_labels: (dataset.input_labels || []).join(', '),
            output_labels: (dataset.output_labels || []).join(', '),
            width: dataset.width ?? null,
            height: dataset.height ?? null,
            channels: dataset.channels ?? 1,
            is_input_only: Boolean(dataset.is_input_only),
            sampleCount: 8,
          }
        : empty,
    );
  }, [open, dataset]);

  const patch = (p) => setForm((f) => ({ ...f, ...p }));
  const isImage = form.ds_type === 'image';

  const split = (s) =>
    String(s || '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

  const payload = useMemo(
    () => ({
      name: form.name.trim(),
      description: form.description.trim(),
      ds_type: form.ds_type,
      num_inputs: Number(form.num_inputs) || 1,
      num_outputs: form.is_input_only ? null : Number(form.num_outputs) || 1,
      input_labels: split(form.input_labels),
      output_labels: form.is_input_only ? [] : split(form.output_labels),
      width: isImage ? Number(form.width) || 8 : null,
      height: isImage ? Number(form.height) || 8 : null,
      channels: isImage ? Number(form.channels) || 1 : 1,
      is_input_only: form.is_input_only,
    }),
    [form, isImage],
  );

  const size = isImage ? (payload.width || 8) * (payload.height || 8) * (payload.channels || 1) : payload.num_inputs;

  async function submit() {
    if (!payload.name) {
      toast.warn('Give the dataset a name.');
      return;
    }
    if (size > 4096) {
      toast.warn('That input vector is huge — keep images at 32×32 or smaller.');
      return;
    }
    setBusy(true);
    try {
      if (editing) {
        await api.updateDataset(dataset.id, payload);
        toast.success(`Dataset “${payload.name}” updated`);
      } else {
        const data = makeSamples({ ...payload, count: Number(form.sampleCount) || 0, size });
        const res = await api.createDataset({ ...payload, data });
        toast.success(`Dataset “${payload.name}” created with ${data.length} samples`);
        onSaved?.(res?.dataset?.id ?? null);
      }
      await refreshDatasets();
      onClose();
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={editing ? 'Edit dataset' : 'New dataset'}
      subtitle={
        editing
          ? 'Predefined datasets are read-only — clone one to edit it.'
          : 'Metadata first; you can edit every sample afterwards.'
      }
      icon="database"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" icon={editing ? 'save' : 'plus'} loading={busy} onClick={submit}>
            {editing ? 'Save changes' : 'Create dataset'}
          </Button>
        </>
      }
    >
      <div className="dialog-form">
        <Field label="Name">
          <TextInput value={form.name} onChange={(v) => patch({ name: v })} placeholder="My spirals" autoFocus />
        </Field>
        <Field label="Description" hint="Shown in the picker and on the dataset card.">
          <TextInput
            value={form.description}
            onChange={(v) => patch({ description: v })}
            placeholder="Two interleaved spirals, 2 inputs → 1 class"
          />
        </Field>

        <div className="row wrap" style={{ gap: 10 }}>
          <Field label="Type" className="grow">
            <Select value={form.ds_type} onChange={(v) => patch({ ds_type: v })} options={TYPES} />
          </Field>
          {!isImage && (
            <>
              <Field label="Inputs" style={{ width: 96 }}>
                <NumberInput
                  value={form.num_inputs}
                  min={1}
                  max={512}
                  onChange={(v) => patch({ num_inputs: v })}
                />
              </Field>
              <Field label="Outputs" style={{ width: 96 }}>
                <NumberInput
                  value={form.num_outputs}
                  min={1}
                  max={64}
                  disabled={form.is_input_only}
                  onChange={(v) => patch({ num_outputs: v })}
                />
              </Field>
            </>
          )}
        </div>

        {isImage && (
          <div className="row wrap" style={{ gap: 10 }}>
            <Field label="Width" style={{ width: 90 }}>
              <NumberInput value={form.width ?? 8} min={2} max={64} onChange={(v) => patch({ width: v })} />
            </Field>
            <Field label="Height" style={{ width: 90 }}>
              <NumberInput value={form.height ?? 8} min={2} max={64} onChange={(v) => patch({ height: v })} />
            </Field>
            <Field label="Channels" style={{ width: 90 }}>
              <NumberInput value={form.channels ?? 1} min={1} max={4} onChange={(v) => patch({ channels: v })} />
            </Field>
            <div className="grow" style={{ alignSelf: 'flex-end' }}>
              <Badge mono>
                {size} inputs per sample
              </Badge>
            </div>
          </div>
        )}

        <div className="row wrap" style={{ gap: 10 }}>
          <Field label="Input labels" className="grow" hint="Comma separated.">
            <TextInput value={form.input_labels} onChange={(v) => patch({ input_labels: v })} placeholder="X, Y" />
          </Field>
          <Field label="Output labels" className="grow" hint="Comma separated.">
            <TextInput
              value={form.output_labels}
              onChange={(v) => patch({ output_labels: v })}
              placeholder="Class"
              disabled={form.is_input_only}
            />
          </Field>
        </div>

        <div className="row wrap" style={{ gap: 14, alignItems: 'center' }}>
          <Switch
            checked={form.is_input_only}
            onChange={(v) => patch({ is_input_only: v })}
            label="Input-only dataset"
            tip="No targets — useful for autoencoders and latent exploration."
          />
          {!editing && (
            <Field label="Seed samples" style={{ width: 120 }}>
              <NumberInput
                value={form.sampleCount}
                min={0}
                max={500}
                onChange={(v) => patch({ sampleCount: v })}
              />
            </Field>
          )}
        </div>
        {!editing && (
          <p className="tiny muted">
            {Number(form.sampleCount) > 0
              ? `We'll seed ${form.sampleCount} random samples so you have something to edit right away.`
              : 'Start empty and add samples by hand.'}
          </p>
        )}
      </div>
    </Modal>
  );
}

function makeSamples({ count, size, num_outputs, is_input_only, ds_type }) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const x = Array.from({ length: size }, () => Math.round(Math.random() * 1000) / 1000);
    if (is_input_only) {
      out.push({ x, y: [] });
      continue;
    }
    let y;
    if (num_outputs > 1) {
      y = Array.from({ length: num_outputs }, () => 0);
      y[i % num_outputs] = 1;
    } else {
      y = [ds_type === 'tabular' && i % 2 === 0 ? 1 : 0];
    }
    out.push({ x, y });
  }
  return out;
}
