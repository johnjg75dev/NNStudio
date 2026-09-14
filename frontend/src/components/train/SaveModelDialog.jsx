import { useEffect, useState } from 'react';
import api from '../../api/client';
import { Button, Field, Modal, TextInput } from '../ui';
import { useCatalog } from '../../state/CatalogContext';
import { useSession } from '../../state/SessionContext';
import { useToast } from '../../state/ToastContext';
import { fmtLoss, fmtPct } from '../../lib/format';

/** Save the active session's trained weights and topology to the user's model library. */
export default function SaveModelDialog({ open, onClose }) {
  const { refreshModels } = useCatalog();
  const snapshot = useSession((s) => s.snapshot);
  const metrics = useSession((s) => s.metrics);
  const toast = useToast();

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    const taskName = snapshot?.func?.label || snapshot?.func_key || 'Model';
    const accStr = metrics.accuracy !== null && metrics.accuracy !== undefined ? ` · ${fmtPct(metrics.accuracy)} acc` : '';
    const lossStr = metrics.loss !== null && metrics.loss !== undefined ? ` · loss ${fmtLoss(metrics.loss)}` : '';
    setName(`${taskName} (Epoch ${metrics.epoch ?? 0}${accStr})`);
    setDescription(`Trained on ${taskName} — ${(snapshot?.topology || []).join('→')} topology${lossStr}`);
  }, [open, snapshot, metrics]);

  async function save() {
    const label = name.trim();
    if (!label) {
      toast.warn('Give the model a name first.');
      return;
    }
    setBusy(true);
    try {
      await api.saveModel({
        name: label,
        description: description.trim(),
      });
      await refreshModels();
      toast.success(`Saved “${label}” to your library`);
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
      title="Save model to library"
      subtitle="Saves the live network weights, architecture, and training history so you can reload or export later."
      icon="save"
      size="narrow"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" icon="save" onClick={save} loading={busy}>
            Save model
          </Button>
        </>
      }
    >
      <div className="col" style={{ gap: 12 }}>
        <Field label="Model name">
          <TextInput value={name} onChange={setName} placeholder="XOR — 4 hidden, 200 epochs" autoFocus />
        </Field>
        <Field label="Description" hint="Shown in your model library.">
          <TextInput
            value={description}
            onChange={setDescription}
            placeholder="Notes about this training run..."
          />
        </Field>
      </div>
    </Modal>
  );
}
