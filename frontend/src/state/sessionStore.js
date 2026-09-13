/**
 * state/sessionStore.js — the training session.
 *
 * A tiny observable store rather than context-per-value: the training loop
 * mutates state ~60×/s and React re-renders must stay cheap. Components read
 * slices with `useSession(selector)`; canvases subscribe to frame updates
 * directly and repaint without re-rendering the tree.
 */
import api from '../api/client';
import { makeLayer, serialiseLayer } from '../lib/layers';
import { annotateSamples } from '../lib/samples';
import { uid } from '../lib/format';

export const DEFAULT_VIZ = {
  showLabels: true,
  showActivations: true,
  showBias: false,
  showGradients: false,
  showDeadWeights: false,
  showImageIO: false,
  imageIn: [28, 28, 1], // manual image I/O dimensions [w, h, channels]
  imageOut: [10, 1, 1],
};

export const DEFAULT_CONFIG = {
  archKey: 'mlp',
  funcKey: 'xor',
  dsId: '',
  inputs: 2,
  outputs: 1,
  layers: [makeLayer('dense', { neurons: 4, activation: 'tanh' })],
  activation: 'tanh',
  optimizer: 'adam',
  loss: 'bce',
  lr: 0.01,
  weightDecay: 0,
  steps: 10,
};

/** Config keys that require a rebuild when they change. */
const STRUCTURAL_KEYS = [
  'archKey',
  'funcKey',
  'dsId',
  'inputs',
  'outputs',
  'layers',
  'optimizer',
  'loss',
  'weightDecay',
  'activation',
];

const initialState = {
  booted: false,
  status: 'idle', // idle | building | ready | training | paused | error
  statusMessage: '',
  error: null,
  busy: false,

  config: DEFAULT_CONFIG,
  builtConfig: null, // config that produced the current snapshot
  configDirty: false,

  snapshot: null, // full /api/session/snapshot payload
  frame: null, // { layers, activations, lossHistory } — canvas-facing
  metrics: { epoch: 0, loss: null, accuracy: null, params: 0 },
  lossHistory: [],
  samples: [], // evaluated training samples
  running: false,
  startedAt: null,
  stepsRun: 0,

  selectedNode: null,
  focusMode: false,
  viz: DEFAULT_VIZ,
  plot: { showBoundary: true, showPoints: true },

  test: {
    values: [],
    expected: null,
    source: null, // 'manual' | 'sample' | 'latent' | 'playground'
    startLayer: 0,
    endLayer: null,
    output: null,
    activations: null,
  },
  sweep: { ranges: [], results: null },
  latent: null, // { layer, node, value, data }
  playgroundSeed: null, // { x, y } handed to the Playground page
  history: [], // [{ id, action, timestamp, config }]
};

export class SessionStore {
  constructor() {
    this.state = { ...initialState };
    this.listeners = new Set();
    this.frameListeners = new Set();
    this.rafId = null;
    this.inflight = false;
    this.lastMetricsPush = 0;
    this.onNotify = null; // optional hook (toast bridge)
  }

  // ── subscription plumbing ──────────────────────────────────────────
  subscribe = (fn) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  subscribeFrames = (fn) => {
    this.frameListeners.add(fn);
    return () => this.frameListeners.delete(fn);
  };

  getState = () => this.state;

  set(patch, { frame = false } = {}) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((l) => l());
    if (frame) this.frameListeners.forEach((l) => l(this.state));
  }

  notifyFrame() {
    this.frameListeners.forEach((l) => l(this.state));
  }

  notify(message, tone = 'info') {
    this.onNotify?.(message, tone);
  }

  // ── bootstrapping ──────────────────────────────────────────────────
  async boot(preset) {
    if (this.state.booted) return;
    if (preset) this.applyPreset(preset, { silent: true });
    await this.build({ silent: true });
    this.set({ booted: true });
  }

  // ── configuration ──────────────────────────────────────────────────
  setConfig(patch, { markDirty = true } = {}) {
    const config = { ...this.state.config, ...patch };
    const structural = Object.keys(patch).some((k) => STRUCTURAL_KEYS.includes(k));
    const dirty = markDirty && structural ? true : this.state.configDirty;
    this.set({ config, configDirty: dirty });
  }

  setLayers(layers) {
    this.setConfig({ layers });
    this.pushHistory('Edited layer stack');
  }

  setViz(patch) {
    this.set({ viz: { ...this.state.viz, ...patch } }, { frame: true });
  }

  setPlotOptions(patch) {
    this.set({ plot: { ...this.state.plot, ...patch } });
  }

  /** Config payload for /api/session/build. */
  buildPayload(config = this.state.config) {
    const payload = {
      arch_key: config.archKey,
      func_key: config.funcKey,
      inputs: config.inputs,
      outputs: config.outputs,
      layers: config.layers.map(serialiseLayer),
      activation: config.activation,
      optimizer: config.optimizer,
      loss: config.loss,
      lr: config.lr,
      weight_decay: config.weightDecay,
    };
    if (config.dsId) payload.ds_id = Number(config.dsId);
    return payload;
  }

  applyPreset(preset, { silent = false } = {}) {
    const layers = (preset.layers || []).map((l) => makeLayer(l.type, l));
    const config = {
      ...this.state.config,
      archKey: preset.arch_key || 'mlp',
      funcKey: preset.func_key || 'xor',
      dsId: '',
      layers: layers.length ? layers : [makeLayer('dense')],
      optimizer: preset.optimizer || 'adam',
      loss: preset.loss || 'bce',
      lr: preset.lr ?? 0.01,
      weightDecay: preset.weight_decay ?? 0,
      activation: preset.activation || 'tanh',
    };
    this.set({ config, configDirty: true });
    this.pushHistory(`Applied preset “${preset.label}”`);
    if (!silent) this.notify(`Preset “${preset.label}” applied`, 'pos');
  }

  /** Sync input/output widths from the active task or dataset. */
  syncIoDims(meta) {
    if (!meta) return;
    const { config } = this.state;
    const inputs = meta.inputs ?? meta.num_inputs ?? config.inputs;
    const outputs = meta.outputs ?? meta.num_outputs ?? config.outputs;
    if (inputs !== config.inputs || outputs !== config.outputs) {
      this.set({ config: { ...config, inputs, outputs } });
    }
  }

  // ── build / reset ──────────────────────────────────────────────────
  async build({ silent = false } = {}) {
    this.stop({ silent: true });
    this.set({ status: 'building', busy: true, error: null, statusMessage: 'Building network…' });
    try {
      const config = this.state.config;
      const result = await api.build(this.buildPayload(config));
      const snapshot = await api.snapshot();
      if (!snapshot?.built) throw new Error('The server did not build a network.');

      const lossHistory = snapshot.loss_history || [];
      this.set({
        snapshot,
        builtConfig: config,
        configDirty: false,
        status: 'ready',
        statusMessage: 'Ready',
        busy: false,
        error: null,
        lossHistory,
        frame: {
          layers: snapshot.layers,
          activations: snapshot.activations,
          lossHistory,
          topology: snapshot.topology,
        },
        metrics: {
          epoch: snapshot.epoch ?? 0,
          loss: lossHistory.length ? lossHistory[lossHistory.length - 1] : null,
          accuracy: null,
          params: result.param_count ?? snapshot.param_count ?? 0,
        },
        selectedNode: null,
        focusMode: false,
        test: { ...initialState.test },
        latent: null,
      });
      this.setTaskMeta(snapshot.func);
      await this.refreshSamples({ silent: true });
      this.notifyFrame();
      if (!silent) this.notify('Network rebuilt', 'pos');
      return snapshot;
    } catch (e) {
      this.set({
        status: 'error',
        statusMessage: 'Build failed',
        busy: false,
        error: e.message,
      });
      this.notify(e.message, 'neg');
      return null;
    }
  }

  setTaskMeta(func) {
    if (!func) return;
    const inputs = func.inputs ?? this.state.config.inputs;
    const labels = func.input_labels?.length
      ? func.input_labels
      : Array.from({ length: inputs }, (_, i) => `x${i}`);
    const values = labels.map((_, i) => this.state.test.values[i] ?? 0);
    const ranges = labels.map((_, i) => this.state.sweep.ranges[i] || { min: 0, max: 1, step: 0.25 });
    this.set({
      test: { ...this.state.test, values },
      sweep: { ...this.state.sweep, ranges },
    });
  }

  async reset() {
    this.stop({ silent: true });
    this.set({ busy: true, statusMessage: 'Resetting weights…' });
    try {
      await api.reset();
      const snapshot = await api.snapshot();
      this.set({
        snapshot,
        busy: false,
        status: 'ready',
        statusMessage: 'Weights reset',
        lossHistory: [],
        frame: {
          layers: snapshot.layers,
          activations: snapshot.activations,
          lossHistory: [],
          topology: snapshot.topology,
        },
        metrics: { epoch: 0, loss: null, accuracy: null, params: snapshot.param_count ?? 0 },
        stepsRun: 0,
        selectedNode: null,
        focusMode: false,
        latent: null,
      });
      this.pushHistory('Reset weights');
      await this.refreshSamples({ silent: true });
      this.notifyFrame();
      this.notify('Weights re-initialised', 'pos');
    } catch (e) {
      this.set({ busy: false, status: 'error', error: e.message });
      this.notify(e.message, 'neg');
    }
  }

  // ── training loop ──────────────────────────────────────────────────
  async toggleTrain() {
    if (this.state.running) this.stop();
    else await this.start();
  }

  async start() {
    if (this.state.running) return;
    const arch = this.state.snapshot?.arch_key;
    if (!this.state.snapshot?.built) {
      await this.build({ silent: true });
      if (!this.state.snapshot?.built) return;
    }
    if (this.state.configDirty) {
      await this.build({ silent: true });
    }
    this.set({
      running: true,
      status: 'training',
      statusMessage: 'Training',
      error: null,
      startedAt: this.state.startedAt || Date.now(),
    });
    this.loop();
  }

  stop({ silent = false } = {}) {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    const wasRunning = this.state.running;
    this.inflight = false;
    this.set({
      running: false,
      status: wasRunning ? 'paused' : this.state.status,
      statusMessage: wasRunning ? 'Paused' : this.state.statusMessage,
    });
    if (wasRunning && !silent) this.notify('Training paused', 'warn');
  }

  async stepOnce(n = 1) {
    if (this.state.running) return;
    await this.runSteps(n);
  }

  loop = () => {
    if (!this.state.running) return;
    this.runSteps(this.state.config.steps).then(() => {
      if (this.state.running) this.rafId = requestAnimationFrame(this.loop);
    });
  };

  async runSteps(steps) {
    if (this.inflight) return;
    this.inflight = true;
    try {
      const data = await api.trainStep(Math.max(1, steps | 0), this.state.config.lr);
      const lossHistory = data.loss_history || this.state.lossHistory;
      this.set(
        {
          frame: {
            layers: data.layers,
            activations: data.activations,
            lossHistory,
            topology: this.state.snapshot?.topology,
          },
          lossHistory,
          stepsRun: this.state.stepsRun + steps,
          snapshot: this.state.snapshot
            ? {
                ...this.state.snapshot,
                layers: data.layers,
                activations: data.activations,
                epoch: data.epoch,
                loss_history: lossHistory,
              }
            : this.state.snapshot,
        },
        { frame: true },
      );

      // metrics + samples are throttled: React only needs them a few times a second
      const now = performance.now();
      if (now - this.lastMetricsPush > 110) {
        this.lastMetricsPush = now;
        this.set({
          metrics: {
            epoch: data.epoch,
            loss: data.loss,
            accuracy: data.accuracy,
            params: this.state.metrics.params,
          },
        });
      }
      if (data.epoch % 6 === 0) await this.refreshSamples({ silent: true, quiet: true });
    } catch (e) {
      this.stop({ silent: true });
      this.set({ status: 'error', statusMessage: 'Training error', error: e.message });
      this.notify(e.message, 'neg');
    } finally {
      this.inflight = false;
    }
  }

  async refreshSamples({ silent = false, quiet = false } = {}) {
    if (!this.state.snapshot?.built) return;
    try {
      const data = await api.evaluate();
      this.set({
        samples: annotateSamples(data.samples || [], this.state.snapshot?.func),
        metrics: quiet
          ? this.state.metrics
          : {
              ...this.state.metrics,
              loss: data.loss ?? this.state.metrics.loss,
              accuracy: data.accuracy ?? this.state.metrics.accuracy,
            },
      });
    } catch (e) {
      if (!silent) this.notify(e.message, 'neg');
    }
  }

  // ── node selection / focus ─────────────────────────────────────────
  selectNode(node) {
    this.set({ selectedNode: node, focusMode: false }, { frame: true });
  }

  toggleFocus() {
    if (!this.state.selectedNode) return;
    this.set({ focusMode: !this.state.focusMode }, { frame: true });
  }

  clearSelection() {
    this.set({ selectedNode: null, focusMode: false }, { frame: true });
  }

  // ── inference ──────────────────────────────────────────────────────
  setTestValues(values) {
    this.set({ test: { ...this.state.test, values } });
  }

  async predict({ x, y = null, source = 'manual', startLayer = 0, endLayer = null, nodeOverrides = null, quiet = false } = {}) {
    if (!this.state.snapshot?.built) {
      if (!quiet) this.notify('Build a network first.', 'warn');
      return null;
    }
    try {
      const data = await api.predict({ x, startLayer, endLayer, nodeOverrides });
      this.set({
        test: {
          ...this.state.test,
          x,
          expected: y,
          source,
          output: data.output,
          activations: data.activations,
          startLayer,
          endLayer,
        },
        frame: this.state.frame
          ? { ...this.state.frame, activations: data.activations }
          : this.state.frame,
      });
      this.notifyFrame();
      return data;
    } catch (e) {
      if (!quiet) this.notify(e.message, 'neg');
      return null;
    }
  }

  setSweepRanges(ranges) {
    this.set({ sweep: { ...this.state.sweep, ranges } });
  }

  async runSweep() {
    if (!this.state.snapshot?.built) {
      this.notify('Build a network first.', 'warn');
      return null;
    }
    try {
      const data = await api.evaluate({
        ranges: this.state.sweep.ranges,
        startLayer: this.state.test.startLayer,
        endLayer: this.state.test.endLayer,
      });
      this.set({ sweep: { ...this.state.sweep, results: data.samples || [] } });
      return data;
    } catch (e) {
      this.notify(e.message, 'neg');
      return null;
    }
  }

  setLatent(latent) {
    this.set({ latent });
  }

  clearLatent() {
    this.set({ latent: null });
  }

  /**
   * Sweep one hidden neuron across a range while the rest of the network sees
   * the current test input. `value` also runs a single overridden prediction so
   * the output chips stay in sync with the slider.
   */
  async latentSweep({ layer, node, value = null, range = null, x = null } = {}) {
    if (!this.state.snapshot?.built) {
      this.notify('Build a network first.', 'warn');
      return null;
    }
    const input = x || this.state.test.values || this.state.test.x || [];
    const prev = this.state.latent || {};
    const l = layer ?? prev.layer ?? 1;
    const n = node ?? prev.node ?? 0;
    this.set({ busy: true });
    try {
      let data = prev.data;
      if (range) {
        const res = await api.latentSweep({ x: input, layer: l, node: n, range });
        data = res.sweep_data;
      }
      let output = null;
      if (value !== null && value !== undefined) {
        const pred = await api.predict({
          x: input,
          nodeOverrides: { layer: l, node: n, val: Number(value) },
        });
        output = pred.output;
      }
      this.set({
        busy: false,
        latent: { layer: l, node: n, value, data, output, x: input },
        test:
          output !== null
            ? { ...this.state.test, output, source: 'latent', activations: null }
            : this.state.test,
      });
      return data;
    } catch (e) {
      this.set({ busy: false });
      this.notify(e.message, 'neg');
      return null;
    }
  }

  // ── cross-page handoff ────────────────────────────────────────────
  /** Queue an input vector for the Playground page to pick up. */
  sendToPlayground(x, y = null) {
    this.set({ playgroundSeed: { x: [...(x || [])], y } });
  }

  takePlaygroundSeed() {
    const seed = this.state.playgroundSeed;
    if (seed) this.state.playgroundSeed = null;
    return seed || null;
  }

  // ── persistence ────────────────────────────────────────────────────
  async exportModel() {
    if (!this.state.snapshot?.built) {
      this.notify('Build a network first.', 'warn');
      return null;
    }
    try {
      return await api.exportSession();
    } catch (e) {
      this.notify(e.message, 'neg');
      return null;
    }
  }

  async importModel(data) {
    try {
      const result = await api.importSession(data);
      const snapshot = await api.snapshot();
      const lossHistory = snapshot?.loss_history || [];
      this.set({
        snapshot,
        status: 'ready',
        statusMessage: 'Model imported',
        lossHistory,
        frame: {
          layers: snapshot?.layers,
          activations: snapshot?.activations,
          lossHistory,
          topology: snapshot?.topology,
        },
        metrics: {
          epoch: result.epoch ?? 0,
          loss: lossHistory.length ? lossHistory[lossHistory.length - 1] : null,
          accuracy: null,
          params: result.param_count ?? 0,
        },
        config: {
          ...this.state.config,
          archKey: data.arch_key || this.state.config.archKey,
          funcKey: data.func_key || this.state.config.funcKey,
        },
        configDirty: false,
        builtConfig: null,
      });
      this.pushHistory(`Imported model (${(result.topology || []).join('→')})`);
      await this.refreshSamples({ silent: true });
      this.notifyFrame();
      this.notify('Model imported into the session', 'pos');
      return result;
    } catch (e) {
      this.notify(e.message, 'neg');
      return null;
    }
  }

  async loadLibraryModel(id) {
    try {
      await api.loadModel(id);
      const snapshot = await api.snapshot();
      const lossHistory = snapshot?.loss_history || [];
      this.set({
        snapshot,
        status: 'ready',
        statusMessage: 'Model loaded',
        lossHistory,
        frame: {
          layers: snapshot?.layers,
          activations: snapshot?.activations,
          lossHistory,
          topology: snapshot?.topology,
        },
        metrics: {
          epoch: snapshot?.epoch ?? 0,
          loss: lossHistory.length ? lossHistory[lossHistory.length - 1] : null,
          accuracy: null,
          params: snapshot?.param_count ?? 0,
        },
        configDirty: false,
        config: {
          ...this.state.config,
          archKey: snapshot?.arch_key || this.state.config.archKey,
          funcKey: snapshot?.func_key || this.state.config.funcKey,
        },
      });
      this.pushHistory('Loaded a model from your library');
      await this.refreshSamples({ silent: true });
      this.notifyFrame();
      this.notify('Model loaded into the session', 'pos');
      return snapshot;
    } catch (e) {
      this.notify(e.message, 'neg');
      return null;
    }
  }

  // ── modification history (client-side undo trail) ──────────────────
  pushHistory(action) {
    const entry = {
      id: uid('h'),
      action,
      timestamp: new Date().toISOString(),
      config: JSON.parse(JSON.stringify(this.state.config)),
    };
    const history = [entry, ...this.state.history].slice(0, 40);
    this.set({ history });
  }

  async restoreHistory(id) {
    const entry = this.state.history.find((h) => h.id === id);
    if (!entry) return;
    this.set({ config: entry.config, configDirty: true });
    this.pushHistory(`Restored “${entry.action}”`);
    await this.build({ silent: true });
    this.notify('Model state restored', 'pos');
  }
}

export const sessionStore = new SessionStore();
export default sessionStore;
