import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Icon from '../Icon';
import { Badge, Button, EmptyState, SearchInput, Segmented } from '../ui';
import { SevenSegment, SevenSegmentGrid, fourBitToHex } from '../shared/SevenSegment';
import { PixelPreview } from '../shared/PixelPreview';
import { useSession, useSessionActions, useSessionStore } from '../../state/SessionContext';
import { fmtNum, fmtPct, truncateArray } from '../../lib/format';
import { matchesTarget, summarise } from '../../lib/samples';

const PAGE = 12;

/** I/O panel: the sample browser plus one-shot predictions. */
export default function IOPanel() {
  const store = useSessionStore();
  const actions = useSessionActions();
  const navigate = useNavigate();

  const snapshot = useSession((s) => s.snapshot);
  const samples = useSession((s) => s.samples);
  const test = useSession((s) => s.test);
  const viz = useSession((s) => s.viz);
  const config = useSession((s) => s.config);
  const busy = useSession((s) => s.busy);

  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);

  const func = snapshot?.func || null;
  const io = useIoShape();
  const isSeg7 = config.funcKey === 'seg7' || func?.key === 'seg7';
  const isImage = io.isImage || viz.showImageIO;
  const outShape = io.output || io.input;
  const stats = useMemo(() => summarise(samples), [samples]);

  const rows = useMemo(() => {
    let out = samples;
    if (filter === 'wrong') out = out.filter((s) => !s.correct);
    else if (filter === 'right') out = out.filter((s) => s.correct);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      out = out.filter((s) => (s.label || '').toLowerCase().includes(q) || truncateArray(s.x, 99).join(' ').includes(q));
    }
    return out;
  }, [samples, filter, search]);

  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const view = rows.slice(page * PAGE, page * PAGE + PAGE);

  if (!snapshot?.built) {
    return (
      <EmptyState
        icon="network"
        title="Nothing built yet"
        message="Pick a task in the setup panel and build the network — predictions and the sample browser appear here."
      />
    );
  }

  function sendToPlayground(x, y) {
    store.sendToPlayground(x, y);
    navigate('/playground');
  }

  const showManual = test.output && test.source && test.source !== 'latent';

  return (
    <div className="io">
      <div className="io__toolbar">
        <Segmented
          value={filter}
          onChange={(v) => {
            setFilter(v);
            setPage(0);
          }}
          options={[
            { value: 'all', label: `All ${stats.total}` },
            { value: 'right', label: `Right ${stats.correct}` },
            { value: 'wrong', label: `Wrong ${stats.total - stats.correct}` },
          ]}
        />
        <div className="row" style={{ marginLeft: 'auto', gap: 6 }}>
          <SearchInput value={search} onChange={setSearch} placeholder="Filter…" className="search--sm" />
          <Button size="sm" variant="ghost" icon="reset" loading={busy} onClick={() => actions.refreshSamples()}>
            Re-evaluate
          </Button>
        </div>
      </div>

      {showManual && (
        <div className="io__manual">
          <div className="row wrap" style={{ gap: 6 }}>
            <Badge tone="violet">{test.source === 'playground' ? 'from the playground' : 'single forward pass'}</Badge>
            {test.expected && matchesTarget(test.output, test.expected) ? (
              <Badge tone="pos">matches target</Badge>
            ) : test.expected ? (
              <Badge tone="neg">differs from target</Badge>
            ) : null}
          </div>
          <div className="row wrap" style={{ gap: 10, marginTop: 8 }}>
            {isImage ? (
              <>
                <figure className="pixel-figure">
                  <figcaption className="mono tiny muted">network says</figcaption>
                  <PixelPreview data={test.output} shape={outShape} small={false} />
                </figure>
                {test.expected && (
                  <figure className="pixel-figure">
                    <figcaption className="mono tiny muted">expected</figcaption>
                    <PixelPreview data={test.expected} shape={outShape} small={false} />
                  </figure>
                )}
              </>
            ) : isSeg7 ? (
              <div className="row" style={{ gap: 12, alignItems: 'flex-end' }}>
                <SevenSegment values={test.output || []} size={44} />
                {test.expected && (
                  <>
                    <span className="tiny muted">expected</span>
                    <SevenSegment values={test.expected || []} size={44} />
                  </>
                )}
              </div>
            ) : (
              <div className="row wrap" style={{ gap: 4 }}>
                {(test.output || []).map((v, i) => (
                  <span key={i} className="io__chip mono">
                    {func?.output_labels?.[i] ? `${func.output_labels[i]} ` : `y${i} `}
                    {fmtNum(v)}
                  </span>
                ))}
              </div>
            )}
          </div>
          {Array.isArray(test.x) && (
            <div className="tiny muted mt-6">
              input <span className="mono">{truncateArray(test.x, 10).join(' · ')}</span>
            </div>
          )}
        </div>
      )}

      {isSeg7 && !isImage ? (
        <SevenSegmentGrid
          samples={view}
          pick="pred"
          labelFor={(s) => fourBitToHex(s.x)}
          onSelect={(s) => sendToPlayground(s.x, s.y)}
        />
      ) : isImage ? (
        <div className="io__images">
          {view.map((s) => (
            <figure key={s.index} className="pixel-figure">
              <figcaption className="mono tiny">
                {s.correct ? '✓' : '✗'} {s.label}
              </figcaption>
              <div className="row" style={{ gap: 6 }}>
                <div className="center">
                  <PixelPreview data={s.pred} shape={outShape} small />
                  <span className="mono xs muted">pred</span>
                </div>
                <div className="center">
                  <PixelPreview data={s.y} shape={outShape} small />
                  <span className="mono xs muted">target</span>
                </div>
              </div>
              <button type="button" className="link tiny" onClick={() => sendToPlayground(s.x, s.y)}>
                open in playground
              </button>
            </figure>
          ))}
          {!view.length && <div className="tiny muted center">no samples match this filter</div>}
        </div>
      ) : (
        <div className="table-scroll">
          <table className="table table--io">
            <thead>
              <tr>
                <th>#</th>
                <th>Inputs</th>
                <th>Expected</th>
                <th>Predicted</th>
                <th className="center">✓</th>
                <th aria-label="actions" />
              </tr>
            </thead>
            <tbody>
              {view.map((s) => (
                <tr key={s.index} className={s.correct ? '' : 'row--wrong'}>
                  <td className="mono muted">{s.index}</td>
                  <td className="mono tiny truncate" title={(s.x || []).map(fmtNum).join(', ')}>
                    {truncateArray(s.x, 8).join(' · ')}
                  </td>
                  <td className="mono tiny truncate" title={(s.y || []).map(fmtNum).join(', ')}>
                    {truncateArray(s.y, 6).join(' · ')}
                  </td>
                  <td className="mono tiny truncate" title={(s.pred || []).map(fmtNum).join(', ')}>
                    {truncateArray(s.pred, 6).join(' · ')}
                  </td>
                  <td className="center">
                    {s.correct ? (
                      <Icon name="check" className="accent pos" size={14} />
                    ) : (
                      <Icon name="close" className="accent neg" size={14} />
                    )}
                  </td>
                  <td>
                    <div className="row" style={{ gap: 2 }}>
                      <Button
                        size="xs"
                        variant="ghost"
                        iconOnly
                        title="Run this input through the network"
                        onClick={() => actions.predict({ x: s.x, y: s.y, source: 'sample' })}
                      >
                        <Icon name="bolt" size={12} />
                      </Button>
                      <Button
                        size="xs"
                        variant="ghost"
                        iconOnly
                        title="Open in the playground"
                        onClick={() => sendToPlayground(s.x, s.y)}
                      >
                        <Icon name="arrowRight" size={12} />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {!view.length && (
                <tr>
                  <td colSpan={6} className="center tiny muted">
                    No samples match this filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <footer className="io__foot">
        <div className="tiny muted">
          {rows.length} of {stats.total} sample{stats.total === 1 ? '' : 's'}
          {stats.accuracy !== null && (
            <>
              {' · '}
              <span className={stats.accuracy >= 0.999 ? 'pos' : ''}>{fmtPct(stats.accuracy)} correct</span>
            </>
          )}
        </div>
        {pages > 1 && (
          <div className="row" style={{ gap: 4 }}>
            <Button size="xs" variant="ghost" iconOnly disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
              <Icon name="chevronLeft" />
            </Button>
            <span className="mono tiny muted">
              {page + 1}/{pages}
            </span>
            <Button size="xs" variant="ghost" iconOnly disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>
              <Icon name="chevronRight" />
            </Button>
          </div>
        )}
      </footer>
    </div>
  );
}
