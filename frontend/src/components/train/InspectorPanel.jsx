import { useCallback, useState } from 'react';
import IOPanel from './IOPanel';
import NodePanel from './NodePanel';
import LatentPanel from './LatentPanel';
import WeightsPanel from './WeightsPanel';
import HistoryPanel from './HistoryPanel';
import Plot2DCanvas from '../canvas/Plot2DCanvas';
import { EmptyState, Tabs } from '../ui';
import { useSession } from '../../state/SessionContext';

const TABS = [
  { value: 'io', label: 'Samples', icon: 'table' },
  { value: 'node', label: 'Neuron', icon: 'cursor' },
  { value: 'latent', label: 'Latent', icon: 'wave' },
  { value: 'weights', label: 'Weights', icon: 'grid' },
  { value: 'plot', label: 'Plot', icon: 'chart' },
  { value: 'history', label: 'History', icon: 'history' },
];

/** Right column: everything you can inspect about the current network. */
export default function InspectorPanel() {
  const [tab, setTab] = useState('io');
  const [latentNode, setLatentNode] = useState(null);
  const selectedNode = useSession((s) => s.selectedNode);
  const built = useSession((s) => s.snapshot?.built);
  const inputs = useSession((s) => s.config.inputs);

  const exploreLatent = useCallback((node) => {
    setLatentNode(node);
    setTab('latent');
  }, []);

  const plotAvailable = built && inputs <= 2;

  return (
    <div className="inspector">
      <Tabs
        value={tab}
        onChange={setTab}
        tabs={TABS.map((t) => ({
          ...t,
          count: t.value === 'node' && selectedNode ? `c${selectedNode.layer}·n${selectedNode.idx}` : undefined,
          title: t.value === 'plot' && !plotAvailable ? 'Needs a task with 1 or 2 inputs' : undefined,
        }))}
        fill
      />
      <div className="inspector__body">
        {tab === 'io' && <IOPanel />}
        {tab === 'node' && <NodePanel onExploreLatent={exploreLatent} />}
        {tab === 'latent' && <LatentPanel initialNode={latentNode} />}
        {tab === 'weights' && <WeightsPanel />}
        {tab === 'plot' &&
          (plotAvailable ? (
            <div className="plot-tab">
              <Plot2DCanvas />
              <p className="tiny muted">
                The shaded region is the network's decision over the input plane, computed by asking the
                server for predictions on a grid. Training samples are overlaid: ✓ where the network is
                right, ✗ where it is not.
              </p>
            </div>
          ) : (
            <EmptyState
              icon="chart"
              title="Needs a 1- or 2-input task"
              message={`This plot draws the network's response across its input space, which only works with one or two inputs. Your current task has ${inputs}.`}
            />
          ))}
        {tab === 'history' && <HistoryPanel />}
      </div>
    </div>
  );
}
