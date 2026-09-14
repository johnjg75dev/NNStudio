import { useEffect, useRef, useState } from 'react';
import Icon from '../Icon';
import { Badge, Button, Metric, Segmented } from '../ui';
import NetworkCanvas from '../canvas/NetworkCanvas';
import ArchDiagramCanvas from '../canvas/ArchDiagramCanvas';
import LossChartCanvas from '../canvas/LossChartCanvas';
import { useCatalog } from '../../state/CatalogContext';
import { useSession, useSessionActions, useSessionStore } from '../../state/SessionContext';
import { fmtCompact, fmtInt, fmtLoss, fmtPct } from '../../lib/format';

/** Centre column: metrics, the live canvas and the loss curve. */
export default function StagePanel() {
  const store = useSessionStore();
  const actions = useSessionActions();
  const catalog = useCatalog();

  const [view, setView] = useState('network');
  const config = useSession((s) => s.config);
  const snapshot = useSession((s) => s.snapshot);
  const metrics = useSession((s) => s.metrics);
  const running = useSession((s) => s.running);
  const status = useSession((s) => s.status);
  const selectedNode = useSession((s) => s.selectedNode);
  const focusMode = useSession((s) => s.focusMode);
  const stepsRun = useSession((s) => s.stepsRun);

  const rate = useStepsPerSecond(running);
  const architecture = catalog.architectureByKey[config.archKey];
  const func = snapshot?.func || catalog.functionByKey[config.funcKey];

  // Auto-switch to the blueprint when the architecture is diagram-only.
  useEffect(() => {
    if (architecture && architecture.trainable === false) setView('blueprint');
  }, [architecture]);

  const trainStep = () => actions.stepOnce(config.steps);

  return (
    <div className="stage">
      <div className="stage__metrics">
        <div className="metrics grow">
          <Metric label="Epoch" value={fmtInt(metrics.epoch)} tone="accent" />
          <Metric label="Loss" value={fmtLoss(metrics.loss)} />
          <Metric
            label="Accuracy"
            value={metrics.accuracy === null || metrics.accuracy === undefined ? '—' : fmtPct(metrics.accuracy)}
            tone={metrics.accuracy >= 0.999 ? 'pos' : ''}
          />
          <Metric label="Parameters" value={fmtCompact(metrics.params)} />
          <Metric label="Steps" value={fmtCompact(stepsRun)} />
          <Metric label="Rate" value={running ? `${fmtCompact(rate)}/s` : '—'} />
        </div>

        <div className="row" style={{ gap: 6 }}>
          <Segmented
            value={view}
            onChange={setView}
            options={[
              { value: 'network', label: 'Network', icon: 'network' },
              { value: 'blueprint', label: 'Blueprint', icon: 'cube' },
            ]}
          />
          <Button
            size="sm"
            variant="ghost"
            iconOnly
            title="Run one step"
            onClick={trainStep}
            disabled={running || !snapshot?.built}
          >
            <Icon name="step" />
          </Button>
        </div>
      </div>

      <div className="stage__canvas">
        {view === 'network' ? (
          <>
            <NetworkCanvas snapshot={snapshot} archTrainable={architecture?.trainable !== false} />
            <div className="canvas-frame__badge">
              <Badge tone="accent">{architecture?.label || config.archKey}</Badge>
              <Badge mono>{func?.label || config.funcKey}</Badge>
              {snapshot?.topology && <Badge mono>[{snapshot.topology.join('→')}]</Badge>}
              {selectedNode && (
                <Badge tone={focusMode ? 'warn' : 'violet'}>
                  {focusMode ? 'tracing influences' : `L${selectedNode.layer} · N${selectedNode.idx}`}
                </Badge>
              )}
              {status === 'building' && <Badge tone="warn">building…</Badge>}
            </div>
          </>
        ) : (
          <>
            <ArchDiagramCanvas archKey={architecture?.diagram_type || config.archKey} />
            <div className="canvas-frame__badge">
              <Badge tone="violet">blueprint</Badge>
              <Badge>{architecture?.label || config.archKey}</Badge>
            </div>
          </>
        )}
      </div>

      <div className="stage__foot">
        <LossChartCanvas height={120} />
      </div>
    </div>
  );
}

/** Throughput estimate: steps completed over the last second. */
function useStepsPerSecond(running) {
  const store = useSessionStore();
  const [rate, setRate] = useState(0);
  const sample = useRef({ steps: 0, t: performance.now() });

  useEffect(() => {
    if (!running) {
      setRate(0);
      return undefined;
    }
    sample.current = { steps: store.getState().stepsRun, t: performance.now() };
    const id = setInterval(() => {
      const now = performance.now();
      const prev = sample.current;
      const dt = (now - prev.t) / 1000;
      const ds = store.getState().stepsRun - prev.steps;
      if (dt > 0.05) {
        setRate((r) => (r ? r * 0.55 + (ds / dt) * 0.45 : ds / dt));
        sample.current = { steps: store.getState().stepsRun, t: now };
      }
    }, 500);
    return () => clearInterval(id);
  }, [running, store]);

  return Math.round(rate);
}
