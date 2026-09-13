import { useCallback, useEffect, useRef, useState } from 'react';
import { NetworkGraph, drawEmptyState } from '../../lib/networkGraph';
import { palette } from '../../lib/colors';
import { useElementSize } from '../../lib/hooks';
import { useSession, useSessionActions, useSessionStore } from '../../state/SessionContext';
import { useTheme } from '../../state/ThemeContext';
import Icon from '../Icon';

const DPR_CAP = 2;

/**
 * NetworkCanvas — the live network graph.
 * Owns zoom/pan/hover locally; reads weights + activations from the session
 * store imperatively so a training frame never re-renders React.
 */
export default function NetworkCanvas({ snapshot, archTrainable, onNodeClick }) {
  const store = useSessionStore();
  const actions = useSessionActions();
  const { theme } = useTheme();

  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const graphRef = useRef(null);
  if (!graphRef.current) graphRef.current = new NetworkGraph();
  const graph = graphRef.current;

  const [zoom, setZoom] = useState(1);
  const [hover, setHover] = useState(null); // { x, y, node, label, value, bias }
  const [dragging, setDragging] = useState(false);
  const dragState = useRef(null);

  const viz = useSession((s) => s.viz);
  const selectedNode = useSession((s) => s.selectedNode);
  const focusMode = useSession((s) => s.focusMode);

  useEffect(() => {
    graph.setOptions(viz);
    paint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viz, theme]);

  useEffect(() => {
    graph.selected = selectedNode;
    graph.focusMode = focusMode;
    paint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedNode, focusMode, theme]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = size;
    if (width < 4 || height < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const state = store.getState();
    const snap = state.snapshot;
    if (!snap || !snap.built) {
      drawEmptyState(ctx, width, height, palette(theme), 'Build a network to see it here');
      return;
    }
    graph.draw(ctx, snap, width, height, theme);
  }, [size.width, size.height, store, graph, theme]);

  // resize / snapshot / theme
  useEffect(() => {
    paint();
  }, [paint]);

  // live frames from the training loop
  useEffect(() => {
    let queued = false;
    const unsub = store.subscribeFrames(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        paint();
      });
    });
    return unsub;
  }, [store, paint]);

  /* ── interaction ── */
  const localPoint = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMove = (e) => {
    const { x, y } = localPoint(e);
    if (dragState.current) {
      const d = dragState.current;
      graph.panBy(x - d.x, y - d.y);
      dragState.current = { x, y, moved: d.moved || Math.abs(x - d.x0) + Math.abs(y - d.y0) > 4 };
      paint();
      return;
    }
    if (!archTrainable) {
      setHover(null);
      return;
    }
    const node = graph.nodeAt(x, y);
    if (!node) {
      setHover(null);
      return;
    }
    const state = store.getState();
    const snap = state.snapshot;
    const acts = snap?.activations || [];
    const fn = snap?.func || {};
    const topo = snap?.topology || [];
    const isInput = node.layer === 0;
    const isOutput = node.layer === topo.length - 1;
    const label = isInput
      ? fn.input_labels?.[node.idx] || `In ${node.idx}`
      : isOutput
        ? fn.output_labels?.[node.idx] || `Out ${node.idx}`
        : `H${node.layer} · N${node.idx}`;
    const value = acts[node.layer]?.[node.idx] ?? 0;
    const bias = node.layer > 0 ? snap?.layers?.[node.layer - 1]?.b?.[node.idx] : undefined;
    setHover({ x, y, node, label, value, bias, banded: node.banded, total: node.column?.total });
  };

  const handleDown = (e) => {
    if (e.button !== 0) return;
    const { x, y } = localPoint(e);
    dragState.current = { x, y, x0: x, y0: y, moved: false };
    setDragging(true);
  };

  const handleUp = (e) => {
    const wasDrag = dragState.current?.moved;
    dragState.current = null;
    setDragging(false);
    if (wasDrag || !archTrainable) return;

    const { x, y } = localPoint(e);
    const expandLayer = graph.expandToggleAt(x, y);
    if (expandLayer !== null) {
      graph.toggleExpanded(expandLayer);
      paint();
      return;
    }
    const node = graph.nodeAt(x, y);
    actions.selectNode(node);
    onNodeClick?.(node);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const { x, y } = localPoint(e);
    const next = graph.zoomBy(-e.deltaY, { x, y });
    setZoom(next);
    paint();
  };

  const handleLeave = () => {
    setHover(null);
    dragState.current = null;
    setDragging(false);
  };

  const zoomTo = (delta) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    const anchor = rect ? { x: rect.width / 2, y: rect.height / 2 } : undefined;
    setZoom(graph.zoomBy(delta, anchor));
    paint();
  };

  const resetView = () => {
    graph.resetView();
    setZoom(1);
    paint();
  };

  return (
    <div className="canvas-frame" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMove}
        onMouseDown={handleDown}
        onMouseUp={handleUp}
        onMouseLeave={handleLeave}
        onWheel={handleWheel}
        style={{ cursor: dragging ? 'grabbing' : hover ? 'pointer' : 'grab' }}
      />

      <div className="canvas-frame__tools">
        <button className="tool-btn" onClick={() => zoomTo(-1)} title="Zoom out">
          <Icon name="zoomOut" />
        </button>
        <span className="zoom-readout">{Math.round(zoom * 100)}%</span>
        <button className="tool-btn" onClick={() => zoomTo(1)} title="Zoom in">
          <Icon name="zoomIn" />
        </button>
        <span className="tool-sep" />
        <button className="tool-btn" onClick={resetView} title="Reset view">
          <Icon name="fit" />
        </button>
      </div>

      {hover && (
        <div
          className="tooltip"
          style={{
            left: Math.min(hover.x + 16, size.width - 190),
            top: Math.max(6, hover.y - 12),
            position: 'absolute',
          }}
        >
          <b>{hover.banded ? `Layer ${hover.node.layer}` : hover.label}</b>
          {hover.banded ? (
            <div>
              {hover.total} units · collapsed
              <br />
              <span className="tr">click ▸ expand in the header</span>
            </div>
          ) : (
            <div>
              activation <span className="mono">{Number(hover.value).toFixed(4)}</span>
              {hover.bias !== undefined && (
                <>
                  <br />
                  bias <span className="mono">{Number(hover.bias).toFixed(4)}</span>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
