/**
 * lib/layers.js — the layer catalogue used by the layer editor.
 *
 * Mirrors `LAYER_TYPES` in app/core/network.py and the field set the
 * `/api/session/build` endpoint understands.
 */

export const ACTIVATIONS = [
  { key: 'relu', label: 'ReLU' },
  { key: 'leakyrelu', label: 'Leaky ReLU' },
  { key: 'tanh', label: 'Tanh' },
  { key: 'sigmoid', label: 'Sigmoid' },
  { key: 'gelu', label: 'GELU' },
  { key: 'swish', label: 'Swish' },
];

export const OPTIMIZERS = [
  { key: 'sgd', label: 'SGD', hint: 'Simple, sensitive to learning rate.' },
  { key: 'momentum', label: 'SGD + Momentum', hint: 'Smooths oscillations in ravines.' },
  { key: 'rmsprop', label: 'RMSProp', hint: 'Adaptive per-parameter learning rate.' },
  { key: 'adam', label: 'Adam', hint: 'Best all-round default.' },
  { key: 'adamw', label: 'AdamW', hint: 'Adam with decoupled weight decay.' },
];

export const LOSSES = [
  { key: 'mse', label: 'MSE', hint: 'Mean squared error — regression.' },
  { key: 'bce', label: 'Binary Cross-Entropy', hint: 'Best for 0/1 classification.' },
  { key: 'mae', label: 'MAE', hint: 'Mean absolute error — robust to outliers.' },
];

/** Layer types grouped for the "add layer" catalogue. */
export const LAYER_CATALOG = [
  {
    group: 'Core',
    items: [
      {
        type: 'dense',
        icon: 'brain',
        name: 'Dense',
        blurb: 'Every input connects to every output. The workhorse of MLPs.',
        fields: [
          { id: 'neurons', label: 'Neurons', kind: 'number', def: 4, min: 1, max: 512 },
          { id: 'activation', label: 'Activation', kind: 'select', def: 'tanh', options: ACTIVATIONS },
        ],
      },
      {
        type: 'dropout',
        icon: 'filter',
        name: 'Dropout',
        blurb: 'Randomly zeroes activations during training to fight overfitting.',
        fields: [
          { id: 'rate', label: 'Rate', kind: 'number', def: 0.5, min: 0, max: 0.9, step: 0.05 },
        ],
      },
      {
        type: 'batchnorm',
        icon: 'sliders',
        name: 'BatchNorm',
        blurb: 'Normalises layer outputs to stabilise and speed up training.',
        fields: [],
      },
      {
        type: 'layernorm',
        icon: 'sliders',
        name: 'LayerNorm',
        blurb: 'Normalises across features — the Transformer default.',
        fields: [{ id: 'eps', label: 'Epsilon', kind: 'number', def: 0.00001, step: 0.000001 }],
      },
    ],
  },
  {
    group: 'Vision',
    items: [
      {
        type: 'conv2d',
        icon: 'grid',
        name: 'Conv2D',
        blurb: 'Sliding kernels that detect local spatial features.',
        fields: [
          { id: 'out_channels', label: 'Filters', kind: 'number', def: 16, min: 1, max: 256 },
          { id: 'kernel_size', label: 'Kernel', kind: 'number', def: 3, min: 1, max: 7 },
          { id: 'stride', label: 'Stride', kind: 'number', def: 1, min: 1, max: 5 },
          { id: 'padding', label: 'Padding', kind: 'number', def: 1, min: 0, max: 5 },
          {
            id: 'activation',
            label: 'Activation',
            kind: 'select',
            def: 'relu',
            options: ACTIVATIONS.slice(0, 4),
          },
        ],
      },
      {
        type: 'maxpool2d',
        icon: 'minimize',
        name: 'MaxPool',
        blurb: 'Downsamples by taking the max of each window.',
        fields: [
          { id: 'pool_size', label: 'Pool size', kind: 'number', def: 2, min: 2, max: 4 },
          { id: 'stride', label: 'Stride', kind: 'number', def: 2, min: 1, max: 4 },
        ],
      },
      {
        type: 'flatten',
        icon: 'table',
        name: 'Flatten',
        blurb: 'Reshapes a spatial volume into a 1-D feature vector.',
        fields: [],
      },
    ],
  },
  {
    group: 'Sequence',
    items: [
      {
        type: 'lstm',
        icon: 'history',
        name: 'LSTM',
        blurb: 'Gated recurrent cell with long-range memory.',
        fields: [
          { id: 'hidden_size', label: 'Hidden size', kind: 'number', def: 64, min: 8, max: 512 },
          { id: 'return_sequences', label: 'Return sequences', kind: 'checkbox', def: true },
        ],
      },
      {
        type: 'simple_rnn',
        icon: 'reset',
        name: 'Simple RNN',
        blurb: 'Basic recurrent layer — cheap but forgetful.',
        fields: [
          { id: 'hidden_size', label: 'Hidden size', kind: 'number', def: 64, min: 8, max: 512 },
          { id: 'return_sequences', label: 'Return sequences', kind: 'checkbox', def: true },
        ],
      },
      {
        type: 'embedding',
        icon: 'list',
        name: 'Embedding',
        blurb: 'Maps discrete token ids to dense vectors.',
        fields: [
          { id: 'vocab_size', label: 'Vocab size', kind: 'number', def: 1000, min: 10, max: 100000 },
          { id: 'embed_dim', label: 'Embed dim', kind: 'number', def: 64, min: 8, max: 512 },
        ],
      },
      {
        type: 'multihead_attention',
        icon: 'target',
        name: 'Attention',
        blurb: 'Multi-head self attention (Q·Kᵀ/√d → softmax → ·V).',
        fields: [
          { id: 'num_heads', label: 'Heads', kind: 'number', def: 8, min: 1, max: 32 },
          { id: 'd_model', label: 'd_model', kind: 'number', def: 512, min: 8, max: 2048 },
        ],
      },
      {
        type: 'positional_encoding',
        icon: 'shuffle',
        name: 'Pos. Encoding',
        blurb: 'Injects sinusoidal position information into a sequence.',
        fields: [
          { id: 'd_model', label: 'd_model', kind: 'number', def: 512, min: 8, max: 2048 },
          { id: 'max_len', label: 'Max length', kind: 'number', def: 2048, min: 100, max: 10000 },
        ],
      },
    ],
  },
];

export const LAYER_BY_TYPE = Object.fromEntries(
  LAYER_CATALOG.flatMap((g) => g.items).map((item) => [item.type, item]),
);

export const LAYER_LABELS = {
  dense: 'Dense',
  dropout: 'Dropout',
  batchnorm: 'BatchNorm',
  layernorm: 'LayerNorm',
  conv2d: 'Conv2D',
  maxpool2d: 'MaxPool',
  flatten: 'Flatten',
  lstm: 'LSTM',
  simple_rnn: 'RNN',
  embedding: 'Embed',
  multihead_attention: 'Attention',
  positional_encoding: 'PosEnc',
  input: 'Input',
};

export function layerLabel(type) {
  return LAYER_LABELS[type] || type || 'Layer';
}

/** Build a fresh layer config from the catalogue defaults. */
export function makeLayer(type, overrides = {}) {
  const spec = LAYER_BY_TYPE[type];
  const layer = { type };
  if (spec) {
    for (const f of spec.fields) layer[f.id] = overrides[f.id] ?? f.def;
  }
  return { ...layer, ...overrides, id: overrides.id || cryptoId() };
}

/** Strip editor-only fields and emit the payload the API expects. */
export function serialiseLayer(layer) {
  const out = { type: layer.type };
  const spec = LAYER_BY_TYPE[layer.type];
  if (spec) {
    for (const f of spec.fields) {
      if (layer[f.id] !== undefined && layer[f.id] !== null) out[f.id] = layer[f.id];
    }
  }
  return out;
}

export function layerSummary(layer) {
  switch (layer.type) {
    case 'dense':
      return `${layer.neurons ?? '?'} × ${layer.activation ?? 'tanh'}`;
    case 'dropout':
      return `p=${layer.rate ?? 0.5}`;
    case 'conv2d':
      return `${layer.out_channels ?? 16}ch · ${layer.kernel_size ?? 3}² · s${layer.stride ?? 1}`;
    case 'maxpool2d':
      return `${layer.pool_size ?? 2}² pool`;
    case 'lstm':
    case 'simple_rnn':
      return `h=${layer.hidden_size ?? 64}`;
    case 'embedding':
      return `${layer.vocab_size ?? 0}×${layer.embed_dim ?? 0}`;
    case 'multihead_attention':
      return `${layer.num_heads ?? 8} heads · d${layer.d_model ?? 0}`;
    case 'positional_encoding':
      return `d${layer.d_model ?? 0} · L${layer.max_len ?? 0}`;
    case 'layernorm':
      return `ε=${layer.eps ?? 1e-5}`;
    default:
      return '';
  }
}

/** Layers that carry no neurons of their own (drawn as utility nodes). */
export const UTILITY_TYPES = [
  'dropout',
  'batchnorm',
  'flatten',
  'layernorm',
  'positional_encoding',
];

let counter = 0;
function cryptoId() {
  counter += 1;
  const rand =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `l${Date.now().toString(36)}${counter}${rand}`;
}
