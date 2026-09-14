import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Icon from '../components/Icon';
import InputControls from '../components/playground/InputControls';
import OutputDisplay from '../components/playground/OutputDisplay';
import SweepPanel from '../components/playground/SweepPanel';
import { Badge, Button, Card, EmptyState } from '../components/ui';
import { useSession, useSessionStore } from '../state/SessionContext';
import { useIoShape } from '../lib/ioShape';

/**
 * PlaygroundPage — run the current network by hand.
 *
 * Everything shares the live training session, so a model trained on the Train
 * page can be probed here immediately (and samples can be sent over with one
 * click from the sample browser).
 */
export default function PlaygroundPage() {
  const store = useSessionStore();
  const built = useSession((s) => s.snapshot?.built);
  const snapshot = useSession((s) => s.snapshot);
  const test = useSession((s) => s.test);
  const io = useIoShape();
  const [mode, setMode] = useState('numbers');

  // Pick up an input handed over from the Train page sample browser.
  useEffect(() => {
    const seed = store.takePlaygroundSeed();
    if (!seed || !store.getState().snapshot?.built) return;
    store.setTestValues(seed.x);
    store.predict({ x: seed.x, y: seed.y, source: 'playground' });
  }, [store]);

  if (!built) {
    return (
      <div className="page">
        <EmptyState
          icon="beaker"
          title="No network in the session"
          message="The playground drives whatever network is currently built. Head to the Train page, pick a task and build it — then come back to poke it by hand."
          action={
            <Button as={Link} to="/train" variant="primary" icon="build">
              Go to the studio
            </Button>
          }
        />
      </div>
    );
  }

  const func = snapshot?.func || null;

  return (
    <div className="page playground">
      <div className="playground__grid">
        <Card
          title="Input"
          subtitle={
            func
              ? `${func.label} — ${func.inputs} input${func.inputs === 1 ? '' : 's'}${
                  func.input_labels?.length ? ` (${func.input_labels.slice(0, 6).join(', ')}${func.input_labels.length > 6 ? '…' : ''})` : ''
                }`
              : 'Drive the network by hand'
          }
          actions={
            io.isImage ? (
              <Badge tone="violet">
                {io.input ? `${io.input[0]}×${io.input[1]} pixels` : 'image output'}
              </Badge>
            ) : null
          }
        >
          <InputControls mode={mode} onModeChange={setMode} />
        </Card>

        <Card
          title="Output"
          subtitle="A single forward pass through the current weights"
          actions={
            test.output ? (
              <Badge tone="pos" mono>
                {(test.output || []).length} value{(test.output || []).length === 1 ? '' : 's'}
              </Badge>
            ) : null
          }
        >
          <OutputDisplay />
        </Card>
      </div>

      <Card
        title="Grid sweep"
        subtitle="Evaluate a whole range of inputs at once and watch the response surface"
        actions={
          <span className="row" style={{ gap: 6 }}>
            <Badge mono>sweep</Badge>
            <span className="tiny muted">
              <Icon name="info" size={12} /> computed server-side
            </span>
          </span>
        }
      >
        <SweepPanel />
      </Card>
    </div>
  );
}
