import { useEffect, useState } from 'react';
import api from '../../api/client';
import { Button, Field, Modal, TextInput } from '../ui';
import { useCatalog } from '../../state/CatalogContext';
import { useToast } from '../../state/ToastContext';
import { serialiseLayer } from '../../lib/layers';

/** Save the current setup as a reusable preset on the account. */
export default function SavePresetDialog({ open, onClose, config, taskLabel }) {
  const { refreshRegistry } = useCatalog();
  const toast = useToast();
  const [label, setLabel] = useState('');
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLabel(taskLabel ? `${taskLabel} — my setup` : 'My preset');
    setDescription(
      `${config.layers.length} hidden layer(s) · ${config.optimizer} · ${config.loss} · lr ${config.lr}`,
    );
  }, [open, taskLabel, config.layers.length, config.optimizer, config.loss, config.lr]);

  async function save() {
    if (!label.trim()) {
      toast.warn('Give the preset a name first.');
      return;
    }
    setBusy(true);
    try {
      await api.savePreset({
        label: label.trim(),
        description: description.trim(),
        arch_key: config.archKey,
        func_key: config.funcKey,
        layers: config.layers.map(serialiseLayer),
        activation: config.activation,
        optimizer: config.optimizer,
        loss: config.loss,
        lr: config.lr,
        weight_decay: config.weightDecay,
      });
      await refreshRegistry();
      toast.success(`Preset “${label.trim()}” saved`);
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
      title="Save this setup as a preset"
      subtitle="Presets store the task, layer stack and hyperparameters — not the weights."
      icon="save"
      size="narrow"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" icon="save" onClick={save} loading={busy}>
            Save preset
          </Button>
        </>
      }
    >
      <div className="col" style={{ gap: 12 }}>
        <Field label="Name">
          <TextInput value={label} onChange={setLabel} placeholder="My XOR setup" autoFocus />
        </Field>
        <Field label="Description" hint="Shown in the preset gallery.">
          <TextInput
            value={description}
            onChange={setDescription}
            placeholder="What makes this setup interesting?"
          />
        </Field>
      </div>
    </Modal>
  );
}
