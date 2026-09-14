import { useEffect, useState } from 'react';
import SetupPanel from '../components/train/SetupPanel';
import StagePanel from '../components/train/StagePanel';
import InspectorPanel from '../components/train/InspectorPanel';
import Icon from '../components/Icon';
import { Button } from '../components/ui';
import { useSession } from '../state/SessionContext';

/**
 * TrainPage — the studio: setup on the left, the live network in the middle,
 * inspection on the right. Columns collapse on narrow screens via a local
 * focus toggle so the canvas always gets room to breathe.
 */
export default function TrainPage() {
  const [focus, setFocus] = useState(null); // null = all three columns
  const error = useSession((s) => s.error);
  const statusMessage = useSession((s) => s.statusMessage);
  const dirty = useSession((s) => s.configDirty);
  const built = useSession((s) => s.snapshot?.built);

  // Escape returns to the three-column layout.
  useEffect(() => {
    if (!focus) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') setFocus(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focus]);

  return (
    <div className="train-page-wrapper">
      <div className="train-page__switcher" data-has-focus={focus ? 'true' : 'false'}>
        <div className="train-page__tabs">
          <button
            type="button"
            className={`train-page__tab ${focus === 'setup' ? 'train-page__tab--active' : ''}`}
            onClick={() => setFocus(focus === 'setup' ? null : 'setup')}
          >
            <Icon name="sliders" size={13} />
            <span>Setup</span>
          </button>
          <button
            type="button"
            className={`train-page__tab ${focus === 'stage' ? 'train-page__tab--active' : ''}`}
            onClick={() => setFocus(focus === 'stage' ? null : 'stage')}
          >
            <Icon name="network" size={13} />
            <span>Network</span>
          </button>
          <button
            type="button"
            className={`train-page__tab ${focus === 'inspector' ? 'train-page__tab--active' : ''}`}
            onClick={() => setFocus(focus === 'inspector' ? null : 'inspector')}
          >
            <Icon name="eye" size={13} />
            <span>Inspect</span>
          </button>
        </div>
        {focus && (
          <button
            type="button"
            className="train-page__show-all"
            onClick={() => setFocus(null)}
            title="Show all three panels side-by-side"
          >
            <Icon name="maximize" size={12} />
            <span>Show all</span>
            <kbd>Esc</kbd>
          </button>
        )}
      </div>

      <div className="train-page" data-focus={focus || 'all'}>
        <aside className="train-page__col train-page__col--setup" data-col="setup">
          <ColumnHead
            title="Setup"
            hint="Task, layers, hyperparameters"
            active={focus === 'setup'}
            onFocus={() => setFocus(focus === 'setup' ? null : 'setup')}
          />
          <SetupPanel />
        </aside>

        <section className="train-page__col train-page__col--stage" data-col="stage">
          <ColumnHead
            title="Network"
            hint={dirty ? 'Setup changed — rebuild to apply' : statusMessage || 'Live'}
            tone={dirty ? 'warn' : undefined}
            active={focus === 'stage'}
            onFocus={() => setFocus(focus === 'stage' ? null : 'stage')}
          />
          {error && (
            <div className="banner banner--neg">
              <Icon name="alert" size={14} />
              <span>{error}</span>
            </div>
          )}
          <StagePanel />
        </section>

        <aside className="train-page__col train-page__col--inspector" data-col="inspector">
          <ColumnHead
            title="Inspect"
            hint={built ? 'Samples, neurons, weights' : 'Waiting for a build'}
            active={focus === 'inspector'}
            onFocus={() => setFocus(focus === 'inspector' ? null : 'inspector')}
          />
          <InspectorPanel />
        </aside>

        {focus && (
          <button type="button" className="focus-exit" onClick={() => setFocus(null)}>
            <Icon name="maximize" size={13} /> Show all panels <kbd>Esc</kbd>
          </button>
        )}
      </div>
    </div>
  );
}

function ColumnHead({ title, hint, tone, active, onFocus }) {
  return (
    <header className="col-head">
      <div className="col-head__text">
        <h2>{title}</h2>
        <span className={`col-head__hint ${tone === 'warn' ? 'warn' : 'muted'}`}>{hint}</span>
      </div>
      <Button
        size="xs"
        variant="ghost"
        iconOnly
        title={active ? 'Show all panels' : `Focus the ${title.toLowerCase()} panel`}
        onClick={onFocus}
      >
        <Icon name={active ? 'minimize' : 'maximize'} size={13} />
      </Button>
    </header>
  );
}
