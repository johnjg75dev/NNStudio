import { useState } from 'react';
import Icon from '../Icon';
import WeightHeatmap from './WeightHeatmap';
import ActivationChart from '../canvas/ActivationChart';
import { Badge, EmptyState, Segmented } from '../ui';
import { ACTIVATION_BY_KEY } from '../../lib/activations';
import { useSession } from '../../state/SessionContext';
import { fmtCompact, fmtNum } from '../../lib/format';
import { layerLabel } from '../../lib/layers';

/** Layer activations arrive as a per-neuron list on the server. */
function activationList(layer) {
  const a = layer?.activation;
  if (Array.isArray(a)) return a.length ? a : ['linear'];
  return [a || 'linear'];
}

function activationName(a) {
  const [first] = activationList({ activation: a });
  return ACTIVATION_BY_KEY[first]?.name || first;
}

const NO_LAYERS = [];

/** Weight matrices: heatmaps plus the activation curves each layer uses. */
export default function WeightsPanel() {
  const layers = useSession((s) => s.snapshot?.layers || NO_LAYERS);
  const built = useSession((s) => s.snapshot?.built);
  const params = useSession((s) => s.metrics.params);
  const [mode, setMode] = useState('weights');

  if (!built || !layers.length) {
    return (
      <EmptyState
        icon="grid"
        title="No weights yet"
        message="Build the network to inspect its weight matrices and biases."
      />
    );
  }

  return (
    <div className="weights">
      <div className="row" style={{ gap: 8, marginBottom: 10 }}>
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: 'weights', label: 'Matrices', icon: 'grid' },
            { value: 'activations', label: 'Functions', icon: 'wave' },
          ]}
        />
        <Badge tone="accent" mono style={{ marginLeft: 'auto' }}>
          {fmtCompact(params)} params
        </Badge>
      </div>

      {mode === 'weights' ? (
        <div className="col" style={{ gap: 12 }}>
          {layers.map((l, i) => (
            <div key={i} className="wmat-block">
              <WeightHeatmap layer={l} index={i} />
              <footer className="wmat-block__foot tiny muted">
                <span>
                  {l.is_output
                    ? 'output layer'
                    : `${layerLabel(l.type)} · ${activationName(l.activation)}`}
                </span>
                <span className="mono">
                  {fmtNum(l.W?.length || 0)}×{fmtNum(l.W?.[0]?.length || 0)}
                </span>
              </footer>
            </div>
          ))}
          <p className="tiny muted">
            <Icon name="info" size={12} /> Warm cells are strong positive weights, cool cells are
            strong negative weights. Hover for exact values.
          </p>
        </div>
      ) : (
        <div className="col" style={{ gap: 14 }}>
          {[...new Set(layers.filter((l) => !l.is_output).flatMap((l) => activationList(l)))].map(
            (name) => (
              <div key={name}>
                <header className="sec-head">
                  <h3>{ACTIVATION_BY_KEY[name]?.name || name}</h3>
                  <Badge mono>{ACTIVATION_BY_KEY[name]?.formula || 'identity'}</Badge>
                </header>
                <ActivationChart type={name} height={130} />
                <p className="tiny muted">{ACTIVATION_BY_KEY[name]?.use || 'Passes values through unchanged.'}</p>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}
