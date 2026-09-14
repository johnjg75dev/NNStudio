import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Icon from '../components/Icon';
import ActivationChart from '../components/canvas/ActivationChart';
import ArchDiagramCanvas from '../components/canvas/ArchDiagramCanvas';
import { Badge, Button, Card, HtmlText, Tabs } from '../components/ui';
import { ACTIVATION_INFO } from '../lib/activations';
import { LAYER_CATALOG, LOSSES, OPTIMIZERS } from '../lib/layers';
import { useCatalog } from '../state/CatalogContext';
import { useSessionStore } from '../state/SessionContext';
import { useToast } from '../state/ToastContext';

const SECTIONS = [
  { value: 'loop', label: 'Training loop', icon: 'activity' },
  { value: 'architectures', label: 'Architectures', icon: 'cube' },
  { value: 'activations', label: 'Activations', icon: 'wave' },
  { value: 'optimizers', label: 'Optimizers & loss', icon: 'sliders' },
  { value: 'layers', label: 'Layer types', icon: 'layers' },
  { value: 'glossary', label: 'Glossary', icon: 'book' },
];

const LOOP_STEPS = [
  {
    icon: 'arrowRight',
    title: '1 · Forward pass',
    body: 'Every sample is pushed through the network: each layer multiplies by its weights, adds a bias and squashes the result with an activation function. The last layer produces the prediction.',
  },
  {
    icon: 'target',
    title: '2 · Loss',
    body: 'The prediction is compared with the target using a loss function (MSE, BCE or MAE). One number summarises how wrong the network is across the whole dataset — this is the value the loss chart tracks.',
  },
  {
    icon: 'arrowDown',
    title: '3 · Backward pass',
    body: 'Backpropagation walks the network in reverse, applying the chain rule to work out how much each weight contributed to the error. Those partial derivatives are the gradients (dW) you can see on the canvas.',
  },
  {
    icon: 'sliders',
    title: '4 · Optimizer step',
    body: 'The optimizer nudges every weight against its gradient, scaled by the learning rate. Adam keeps per-parameter momentum; SGD is simpler but needs a well-chosen rate; weight decay shrinks weights slightly each step.',
  },
  {
    icon: 'reset',
    title: '5 · Repeat',
    body: 'One pass over the data is an epoch. Loss should fall and accuracy rise. When loss stops improving while accuracy on unseen data gets worse, the network is memorising rather than learning — that is overfitting.',
  },
];

const GLOSSARY = [
  ['Epoch', 'One full pass over the training data. The studio counts epochs on the metrics strip.'],
  ['Learning rate', 'How far each weight moves against its gradient. Too high overshoots, too low crawls. The slider is logarithmic for that reason.'],
  ['Gradient', 'The slope of the loss with respect to a weight — the direction and size of the correction backpropagation proposes.'],
  ['Overfitting', 'The network memorises the training samples instead of the rule. Training loss keeps falling while accuracy on new data stalls.'],
  ['Underfitting', 'The network is too small or trained too briefly to capture the pattern — both loss and accuracy stay poor.'],
  ['Latent space', 'The internal representation a network builds. Autoencoders compress inputs into a small latent vector and reconstruct from it.'],
  ['Dead neuron', 'A ReLU unit stuck outputting zero because its bias drifted so low nothing can activate it. Leaky ReLU and lower learning rates help.'],
  ['One-hot target', 'A target vector with a single 1 marking the correct class, used for multi-class problems like Iris or MNIST.'],
  ['Decision boundary', 'The surface in input space where the network switches its answer. The Plot tab draws it for one- and two-input tasks.'],
  ['Weight decay', 'L2 regularisation: shrink every weight a little each step so the network prefers simple solutions.'],
];

/** LearnPage — the in-app reference: how training works and what each knob does. */
export default function LearnPage() {
  const [section, setSection] = useState('loop');
  const catalog = useCatalog();
  const store = useSessionStore();
  const toast = useToast();
  const navigate = useNavigate();

  function useArchitecture(key) {
    store.setConfig({ archKey: key }, { markDirty: true });
    store.pushHistory(`Switched architecture to ${key}`);
    navigate('/train');
    toast.info('Architecture selected — press Build in the studio.');
  }

  return (
    <div className="page learn-page">
      <header className="learn-head">
        <div>
          <h1>Learn</h1>
          <p className="lede">
            Everything the studio can do, explained. Read a section, then jump into the network and watch
            the same idea happen live.
          </p>
        </div>
        <Button variant="primary" icon="build" onClick={() => navigate('/train')}>
          Open the studio
        </Button>
      </header>

      <Tabs value={section} onChange={setSection} tabs={SECTIONS} fill />

      {section === 'loop' && (
        <div className="learn-grid">
          <Card title="What happens on every training step" subtitle="The loop the Train button runs">
            <ol className="loop">
              {LOOP_STEPS.map((s) => (
                <li key={s.title} className="loop__step">
                  <span className="loop__icon">
                    <Icon name={s.icon} size={16} />
                  </span>
                  <div>
                    <h3>{s.title}</h3>
                    <p>{s.body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </Card>

          <div className="col" style={{ gap: 12 }}>
            <Card title="Reading the loss chart" subtitle="The curve in the studio footer">
              <ul className="tips">
                <li>
                  <strong>Falling smoothly</strong> — the learning rate is about right. Let it run.
                </li>
                <li>
                  <strong>Spiking or bouncing</strong> — the rate is too high; the optimizer overshoots the
                  minimum each step.
                </li>
                <li>
                  <strong>Almost flat</strong> — the rate is too low, the network is too small, or the task
                  needs a non-linear hidden layer.
                </li>
                <li>
                  <strong>Falls then rises</strong> — overfitting or divergence; stop and reset the weights.
                </li>
              </ul>
              <p className="tiny muted">
                The chart has a log scale toggle — log makes early progress visible, linear shows the true
                shape of convergence.
              </p>
            </Card>

            <Card title="A first experiment" subtitle="Five minutes from zero to a working network">
              <ol className="tips tips--numbered">
                <li>
                  Pick the <strong>XOR Gate</strong> task — two inputs, one output, not linearly separable.
                </li>
                <li>
                  Keep the default <strong>4-neuron tanh</strong> hidden layer and press <strong>Build</strong>.
                </li>
                <li>
                  Press <strong>Train</strong> and watch the loss chart and the neuron colours change at the
                  same time.
                </li>
                <li>
                    Click a hidden neuron, then <strong>Trace influences</strong> to see which inputs drive it.
                </li>
                <li>
                  Open the <strong>Plot</strong> tab: with two inputs you can see the decision boundary form
                  as it learns.
                </li>
              </ol>
            </Card>
          </div>
        </div>
      )}

      {section === 'architectures' && (
        <div className="arch-gallery">
          {(catalog.architectures || []).map((a) => (
            <Card key={a.key} className="arch-card" padded={false}>
              <div className="arch-card__canvas" style={{ '--arch-accent': a.accent_color || 'var(--accent)' }}>
                <ArchDiagramCanvas archKey={a.diagram_type || a.key} height={190} />
              </div>
              <div className="arch-card__body">
                <div className="row wrap" style={{ gap: 6 }}>
                  <h3>{a.label}</h3>
                  <Badge mono>{a.key}</Badge>
                  {a.trainable === false && <Badge tone="warn">reference only</Badge>}
                  {a.is_autoencoder && <Badge tone="violet">latent space</Badge>}
                </div>
                <HtmlText html={a.description} className="tiny" />
                <Button size="sm" variant="ghost" icon="build" onClick={() => useArchitecture(a.key)}>
                  Use in the studio
                </Button>
              </div>
            </Card>
          ))}
          {!catalog.architectures?.length && (
            <p className="tiny muted">Architecture definitions are still loading.</p>
          )}
        </div>
      )}

      {section === 'activations' && (
        <div className="learn-grid learn-grid--acts">
          {ACTIVATION_INFO.map((a) => (
            <Card key={a.key} title={a.name} subtitle={a.formula} className="act-card">
              <ActivationChart type={a.key} height={130} />
              <div className="row wrap" style={{ gap: 6, margin: '8px 0' }}>
                <Badge mono>range {a.range}</Badge>
                <Badge tone="accent">{a.key}</Badge>
              </div>
              <dl className="act-meta">
                <dt>Use it for</dt>
                <dd>{a.use}</dd>
                <dt>Strengths</dt>
                <dd className="pos">{a.pros}</dd>
                <dt>Watch out</dt>
                <dd className="neg">{a.cons}</dd>
              </dl>
            </Card>
          ))}
        </div>
      )}

      {section === 'optimizers' && (
        <div className="learn-grid">
          <Card title="Optimizers" subtitle="How the weights actually move">
            <div className="ref-cards">
              {(catalog.optimizers?.length ? catalog.optimizers : OPTIMIZERS).map((o) => (
                <article key={o.key} className="ref-card">
                  <header className="row" style={{ gap: 6 }}>
                    <h4>{o.label || o.key}</h4>
                    <Badge mono>{o.key}</Badge>
                    {o.lr_range && <Badge tone="accent">{o.lr_range}</Badge>}
                  </header>
                  {o.description && <p className="tiny">{o.description}</p>}
                  {o.hint && <p className="tiny">{o.hint}</p>}
                  {o.pros && (
                    <p className="tiny pos">
                      <Icon name="check" size={12} /> {o.pros}
                    </p>
                  )}
                  {o.cons && (
                    <p className="tiny neg">
                      <Icon name="alert" size={12} /> {o.cons}
                    </p>
                  )}
                </article>
              ))}
            </div>
          </Card>

          <Card title="Loss functions" subtitle="What “wrong” means for a task">
            <div className="ref-cards">
              {LOSSES.map((l) => (
                <article key={l.key} className="ref-card">
                  <header className="row" style={{ gap: 6 }}>
                    <h4>{l.label}</h4>
                    <Badge mono>{l.key}</Badge>
                  </header>
                  <p className="tiny">{l.hint}</p>
                  <p className="tiny muted">{LOSS_NOTES[l.key]}</p>
                </article>
              ))}
            </div>
            <div className="hint-row">
              <Icon name="info" size={14} />
              <span>
                Classification tasks usually want BCE with a sigmoid output; regression wants MSE. Mixing
                them up is the most common reason a network refuses to learn.
              </span>
            </div>
          </Card>
        </div>
      )}

      {section === 'layers' && (
        <div className="layer-catalogue">
          {LAYER_CATALOG.map((group) => (
            <Card key={group.group} title={group.group} subtitle={`${group.items.length} layer types`}>
              <div className="layer-cards">
                {group.items.map((l) => (
                  <article key={l.type} className="layer-card">
                    <header className="row" style={{ gap: 6 }}>
                      <Icon name={l.icon} size={15} />
                      <h4>{l.name}</h4>
                      <Badge mono>{l.type}</Badge>
                    </header>
                    <p className="tiny muted">{l.blurb}</p>
                    {l.fields.length > 0 && (
                      <div className="row wrap" style={{ gap: 4 }}>
                        {l.fields.map((f) => (
                          <span key={f.id} className="chip mono xs">
                            {f.label}
                            {f.def !== undefined ? ` ${f.def}` : ''}
                          </span>
                        ))}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      {section === 'glossary' && (
        <Card title="Glossary" subtitle="The words the interface uses">
          <dl className="glossary">
            {GLOSSARY.map(([term, def]) => (
              <div key={term} className="glossary__row">
                <dt>{term}</dt>
                <dd>{def}</dd>
              </div>
            ))}
          </dl>
        </Card>
      )}
    </div>
  );
}

const LOSS_NOTES = {
  mse: 'Squared error punishes big mistakes hard — gradients grow with the error, which can destabilise classification.',
  bce: 'Log loss for probabilities in [0, 1]. Pairs with a sigmoid output layer and gives strong gradients even when the network is confidently wrong.',
  mae: 'Absolute error treats all mistakes equally, so outliers do not dominate training.',
};
