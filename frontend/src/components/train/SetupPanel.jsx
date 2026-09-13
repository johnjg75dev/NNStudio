import { useMemo, useState } from 'react';
import Icon from '../Icon';
import {
  Accordion,
  Badge,
  Button,
  Field,
  HtmlText,
  Info,
  Select,
  Slider,
  Switch,
} from '../ui';
import LayerStack from './LayerStack';
import AddLayerDialog from './AddLayerDialog';
import TaskPicker from './TaskPicker';
import PresetGallery from './PresetGallery';
import SavePresetDialog from './SavePresetDialog';
import { useCatalog } from '../../state/CatalogContext';
import { useSession, useSessionActions } from '../../state/SessionContext';
import { useConfirm, useToast } from '../../state/ToastContext';
import { LOSSES, OPTIMIZERS, makeLayer } from '../../lib/layers';
import { fmtLr, lrToSlider, sliderToLr } from '../../lib/format';

const TIPS = {
  lr: `<b>Learning rate (η)</b><br>Step size for every weight update.<br><br>
       <span class="tp">Higher:</span> faster early progress<br>
       <span class="tc">Too high:</span> loss explodes or oscillates<br>
       <span class="tc">Too low:</span> crawls towards the optimum<br>
       <span class="tr">Adam 1e-4…1e-2 · SGD 1e-2…1e-1</span>`,
  optimizer: `<b>Optimizer</b><br>Turns gradients into weight updates.<br><br>
       <b>SGD</b> simple, LR-sensitive<br>
       <b>Momentum</b> smooths oscillation<br>
       <b>RMSProp</b> adaptive per-parameter LR<br>
       <b>Adam</b> best all-round default<br>
       <b>AdamW</b> Adam + decoupled weight decay`,
  loss: `<b>Loss function</b><br>What the network minimises.<br><br>
       <b>MSE</b> (ŷ−y)² — regression<br>
       <b>BCE</b> binary cross-entropy — 0/1 outputs<br>
       <b>MAE</b> |ŷ−y| — robust to outliers`,
  decay: `<b>Weight decay</b><br>L2 penalty that keeps weights small.<br><br>
       <span class="tp">Higher:</span> better generalisation<br>
       <span class="tc">Too high:</span> underfitting<br>
       <span class="tr">0 (off) … 0.01 · AdamW default 0.01</span>`,
  steps: `<b>Steps per frame</b><br>Training iterations between redraws.<br><br>
       <span class="tp">Higher:</span> much faster training<br>
       <span class="tc">Higher:</span> less smooth animation<br>
       <span class="tr">1 = watch every update · 50+ = race to convergence</span>`,
  arch: `<b>Architecture</b><br>Picks the blueprint shown on the stage and stored with the model.
       The layer stack below is what actually gets trained.`,
  vizLabels: 'Show the activation value (or task label) inside each neuron.',
  vizActs: 'Colour neurons by activation magnitude — bright means strongly active.',
  vizBias: 'Draw dashed arrows showing each neuron’s learned bias.',
  vizGrad: 'Overlay gradient magnitude on every edge — bright edges are learning fast.',
  vizDead: 'Reveal near-zero weights as faint dotted lines (dead connections).',
};

export default function SetupPanel() {
  const catalog = useCatalog();
  const actions = useSessionActions();
  const toast = useToast();
  const confirm = useConfirm();

  const config = useSession((s) => s.config);
  const dirty = useSession((s) => s.configDirty);
  const busy = useSession((s) => s.busy);
  const running = useSession((s) => s.running);
  const snapshot = useSession((s) => s.snapshot);
  const [taskOpen, setTaskOpen] = useState(false);
  const [presetOpen, setPresetOpen] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [sections, setSections] = useState({ task: true, layers: true, training: false, display: false });

  const taskMeta = useMemo(() => {
    if (config.dsId) {
      const ds = catalog.datasets.find((d) => String(d.id) === String(config.dsId));
      if (ds) {
        return {
          label: ds.name,
          description: ds.description,
          inputs: ds.num_inputs,
          outputs: ds.num_outputs || 1,
          input_labels: ds.input_labels?.length
            ? ds.input_labels
            : Array.from({ length: ds.num_inputs }, (_, i) => `x${i}`),
          output_labels: ds.output_labels?.length
            ? ds.output_labels
            : Array.from({ length: ds.num_outputs || 1 }, (_, i) => `y${i}`),
          kind: 'dataset',
        };
      }
    }
    const fn = catalog.functionByKey[config.funcKey];
    if (fn) {
      return {
        ...fn,
        input_labels: fn.input_labels?.length
          ? fn.input_labels
          : Array.from({ length: fn.inputs }, (_, i) => `x${i}`),
        output_labels: fn.output_labels?.length
          ? fn.output_labels
          : Array.from({ length: fn.outputs }, (_, i) => `y${i}`),
        kind: 'function',
      };
    }
    return null;
  }, [catalog.datasets, catalog.functionByKey, config.dsId, config.funcKey]);

  const architecture = catalog.architectureByKey[config.archKey];
  const topology = snapshot?.topology;

  const toggleSection = (key) => setSections((s) => ({ ...s, [key]: !s[key] }));

  function onAddLayer({ type, values, quantity, at }) {
    const next = [...config.layers];
    const created = Array.from({ length: quantity }, () => makeLayer(type, values));
    next.splice(Math.max(0, Math.min(next.length, at)), 0, ...created);
    actions.setLayers(next);
    toast.success(`Added ${quantity} × ${type} layer${quantity > 1 ? 's' : ''}`);
    setSections((s) => ({ ...s, layers: true }));
  }

  async function onReset() {
    const yes = await confirm({
      title: 'Re-initialise weights?',
      message: 'The topology stays, but every weight is resampled and the epoch counter resets to 0.',
      confirmLabel: 'Reset weights',
      tone: 'warn',
    });
    if (yes) actions.reset();
  }

  return (
    <>
      <div className="panel-scroll">
        <Accordion
          step={1}
          label="Task"
          meta={taskMeta ? `${taskMeta.inputs}→${taskMeta.outputs}` : ''}
          open={sections.task}
          onToggle={() => toggleSection('task')}
          done={!!taskMeta}
        >
          <Field label="Architecture" tip={TIPS.arch}>
            <Select
              value={config.archKey}
              onChange={(v) => actions.setConfig({ archKey: v })}
              options={catalog.architectures.map((a) => ({ value: a.key, label: a.label }))}
            />
          </Field>

          {architecture?.description && (
            <Info tone="accent">
              <HtmlText html={architecture.description} className="rich" />
            </Info>
          )}

          <Field label="Training task">
            <button type="button" className="taskbtn" onClick={() => setTaskOpen(true)}>
              <span className="taskbtn__icon">
                <Icon name={config.dsId ? 'database' : 'cpu'} size={15} />
              </span>
              <span className="taskbtn__body">
                <b className="truncate">{taskMeta?.label || config.funcKey || 'Choose a task'}</b>
                <span className="mono tiny muted">
                  {taskMeta ? `${taskMeta.inputs} inputs → ${taskMeta.outputs} outputs` : '—'}
                </span>
              </span>
              <Icon name="chevronRight" size={14} className="muted" />
            </button>
          </Field>

          {taskMeta?.description && (
            <Info>
              <HtmlText html={taskMeta.description} className="rich" />
            </Info>
          )}

          <div className="btn-group btn-group--fill">
            <Button size="sm" icon="sparkles" onClick={() => setPresetOpen(true)}>
              Presets
            </Button>
            <Button size="sm" variant="ghost" icon="save" onClick={() => setSaveOpen(true)}>
              Save setup
            </Button>
          </div>
        </Accordion>

        <Accordion
          step={2}
          label="Layer stack"
          meta={topology ? topology.join('·') : `${config.layers.length} hidden`}
          open={sections.layers}
          onToggle={() => toggleSection('layers')}
          done={config.layers.length > 0}
        >
          <LayerStack
            layers={config.layers}
            inputs={config.inputs}
            outputs={config.outputs}
            inputLabels={taskMeta?.input_labels}
            outputLabels={taskMeta?.output_labels}
            onChange={(layers) => actions.setLayers(layers)}
          />
          <Button size="sm" variant="outline" block icon="plus" onClick={() => setAddOpen(true)}>
            Add layer
          </Button>
          {topology && (
            <div className="row row--between tiny muted">
              <span className="mono">[{topology.join(' → ')}]</span>
              <Badge mono>{snapshot?.param_count ?? 0} params</Badge>
            </div>
          )}
        </Accordion>

        <Accordion
          step={3}
          label="Training"
          meta={`${config.optimizer} · ${fmtLr(config.lr)}`}
          open={sections.training}
          onToggle={() => toggleSection('training')}
        >
          <Field label="Optimizer" tip={TIPS.optimizer}>
            <Select
              value={config.optimizer}
              onChange={(v) => actions.setConfig({ optimizer: v })}
              options={OPTIMIZERS.map((o) => ({ value: o.key, label: o.label }))}
            />
          </Field>
          <Field
            label="Loss function"
            tip={TIPS.loss}
            hint={LOSSES.find((l) => l.key === config.loss)?.hint}
          >
            <Select
              value={config.loss}
              onChange={(v) => actions.setConfig({ loss: v })}
              options={LOSSES.map((l) => ({ value: l.key, label: l.label }))}
            />
          </Field>

          <Slider
            label="Learning rate"
            tip={TIPS.lr}
            min={-5}
            max={-0.3}
            step={0.05}
            value={lrToSlider(config.lr)}
            format={(v) => fmtLr(sliderToLr(v))}
            onChange={(v) => actions.setConfig({ lr: sliderToLr(v) }, { markDirty: false })}
          />
          <Slider
            label="Weight decay"
            tip={TIPS.decay}
            min={0}
            max={0.05}
            step={0.001}
            value={config.weightDecay}
            format={(v) => Number(v).toFixed(3)}
            onChange={(v) => actions.setConfig({ weightDecay: v })}
          />
          <Slider
            label="Steps per frame"
            tip={TIPS.steps}
            min={1}
            max={200}
            step={1}
            value={config.steps}
            format={(v) => `${v}`}
            onChange={(v) => actions.setConfig({ steps: v }, { markDirty: false })}
          />

          <div className="btn-group btn-group--fill">
            <Button size="sm" variant="primary" icon="build" loading={busy} onClick={() => actions.build()}>
              Rebuild
            </Button>
            <Button size="sm" variant="ghost" icon="reset" onClick={onReset} disabled={busy}>
              Reset weights
            </Button>
          </div>
        </Accordion>

        <Accordion
          step={4}
          label="Display"
          open={sections.display}
          onToggle={() => toggleSection('display')}
        >
          <DisplayOptions />
        </Accordion>
      </div>

      <footer className="setup-foot">
        {dirty && (
          <div className="info info--warn row" style={{ gap: 8, padding: '8px 10px' }}>
            <Icon name="alert" size={14} />
            <span className="grow tiny">Setup changed — rebuild to apply.</span>
            <Button size="sm" variant="warn" onClick={() => actions.build()} loading={busy}>
              Apply
            </Button>
          </div>
        )}
        <div className="row" style={{ gap: 8 }}>
          <Button
            className="btn-transport grow"
            variant={running ? 'default' : 'success'}
            data-running={running ? 'true' : 'false'}
            icon={running ? 'pause' : 'play'}
            onClick={() => actions.toggleTrain()}
          >
            {running ? 'Pause training' : 'Train'}
          </Button>
          <Button
            size="lg"
            variant="ghost"
            iconOnly
            title="Run a single step"
            onClick={() => actions.stepOnce(config.steps)}
            disabled={running || busy}
          >
            <Icon name="step" />
          </Button>
        </div>
        <div className="row row--between tiny muted">
          <span className="row" style={{ gap: 5 }}>
            <kbd>Space</kbd> train
            <kbd>B</kbd> rebuild
            <kbd>R</kbd> reset
          </span>
        </div>
      </footer>

      <TaskPicker
        open={taskOpen}
        onClose={() => setTaskOpen(false)}
        current={{ funcKey: config.funcKey, dsId: config.dsId }}
        onSelect={({ funcKey, dsId, inputs, outputs, meta }) => {
          actions.setConfig({ funcKey, dsId, inputs, outputs });
          actions.syncIoDims(meta);
          actions.pushHistory(`Switched task to ${meta?.label || meta?.name || funcKey}`);
        }}
      />

      <PresetGallery
        open={presetOpen}
        onClose={() => setPresetOpen(false)}
        onApply={(p) => {
          actions.applyPreset(p);
          setSections((s) => ({ ...s, task: true, layers: true }));
        }}
        onSaveCurrent={() => {
          setPresetOpen(false);
          setSaveOpen(true);
        }}
      />

      <SavePresetDialog
        open={saveOpen}
        onClose={() => setSaveOpen(false)}
        config={config}
        taskLabel={taskMeta?.label}
      />

      <AddLayerDialog
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAdd={onAddLayer}
        layerCount={config.layers.length}
      />
    </>
  );
}

function DisplayOptions() {
  const viz = useSession((s) => s.viz);
  const actions = useSessionActions();
  return (
    <div className="col" style={{ gap: 9 }}>
      <Switch
        checked={viz.showActivations}
        onChange={(v) => actions.setViz({ showActivations: v })}
        label="Activation colours"
        tip={TIPS.vizActs}
      />
      <Switch
        checked={viz.showLabels}
        onChange={(v) => actions.setViz({ showLabels: v })}
        label="Node values"
        tip={TIPS.vizLabels}
      />
      <Switch
        checked={viz.showBias}
        onChange={(v) => actions.setViz({ showBias: v })}
        label="Bias arrows"
        tip={TIPS.vizBias}
      />
      <Switch
        checked={viz.showGradients}
        onChange={(v) => actions.setViz({ showGradients: v })}
        label="Gradient overlay"
        tip={TIPS.vizGrad}
      />
      <Switch
        checked={viz.showDeadWeights}
        onChange={(v) => actions.setViz({ showDeadWeights: v })}
        label="Dead weights"
        tip={TIPS.vizDead}
      />
      <hr className="divider" style={{ margin: '2px 0' }} />
      <div className="row row--between">
        <span className="tiny dim">Legend</span>
        <span className="row tiny muted" style={{ gap: 8 }}>
          <span className="row" style={{ gap: 4 }}>
            <i className="swatch" style={{ background: 'var(--pos)' }} /> +w
          </span>
          <span className="row" style={{ gap: 4 }}>
            <i className="swatch" style={{ background: 'var(--neg)' }} /> −w
          </span>
          <span className="row" style={{ gap: 4 }}>
            <i className="swatch" style={{ background: 'var(--accent)' }} /> activation
          </span>
        </span>
      </div>
      <p className="field__hint">
        Edge thickness is proportional to |weight|. Scroll to zoom, drag to pan, click a neuron to
        inspect it.
      </p>
    </div>
  );
}

export { TIPS };
