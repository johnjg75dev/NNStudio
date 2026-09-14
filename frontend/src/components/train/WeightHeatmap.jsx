import { useCallback, useEffect, useRef, useState } from 'react';
import { heatColor, palette } from '../../lib/colors';
import { useElementSize } from '../../lib/hooks';
import { useTheme } from '../../state/ThemeContext';
import { layerLabel } from '../../lib/layers';

const MONO = "'JetBrains Mono', ui-monospace, monospace";

/**
 * WeightHeatmap — one layer's weight matrix as a colour-coded grid.
 * Hovering a cell reports the exact weight and its indices.
 */
export default function WeightHeatmap({ layer, index, maxCells = 64 * 64 }) {
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const [hover, setHover] = useState(null);

  const W = layer?.W;
  const rows = Array.isArray(W) ? W.length : 0;
  const cols = Array.isArray(W?.[0]) ? W[0].length : 0;
  const usable = rows > 0 && cols > 0;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !usable || size.width < 8) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cell = Math.max(2, Math.min(16, Math.floor(size.width / cols)));
    const w = cols * cell;
    const h = rows * cell;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pal = palette(theme);
    const tooMany = rows * cols > maxCells;
    const strideR = tooMany ? Math.ceil(rows / 64) : 1;
    const strideC = tooMany ? Math.ceil(cols / 64) : 1;

    for (let i = 0; i < rows; i += strideR) {
      const row = W[i];
      if (!Array.isArray(row)) continue;
      for (let j = 0; j < cols; j += strideC) {
        const v = Number(row[j]);
        if (!Number.isFinite(v)) continue;
        ctx.fillStyle = heatColor(v, pal, 2);
        ctx.fillRect((j / strideC) * cell, (i / strideR) * cell, cell - 0.6, cell - 0.6);
      }
    }
    if (hover) {
      ctx.strokeStyle = pal.text1;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(
        (hover.j / strideC) * cell,
        (hover.i / strideR) * cell,
        cell - 0.6,
        cell - 0.6,
      );
    }
  }, [W, rows, cols, usable, size.width, theme, hover, maxCells]);

  useEffect(() => {
    draw();
  }, [draw]);

  if (!usable) {
    return (
      <div className="wmat__empty">
        <span className="badge">{layerLabel(layer?.type)}</span>
        <span className="tiny muted">no weight matrix for this layer type</span>
      </div>
    );
  }

  const cell = Math.max(2, Math.min(16, Math.floor(size.width / cols)));
  const strideR = rows * cols > maxCells ? Math.ceil(rows / 64) : 1;
  const strideC = rows * cols > maxCells ? Math.ceil(cols / 64) : 1;

  return (
    <div className="wmat">
      <header className="wmat__head">
        <span className="badge badge--accent">{layerLabel(layer?.type)}</span>
        <span className="mono tiny muted">
          L{index + 1} · {rows}×{cols}
        </span>
        {hover && (
          <span className="mono tiny accent">
            W[{hover.i}][{hover.j}] = {hover.v.toFixed(4)}
          </span>
        )}
      </header>
      <div ref={wrapRef} style={{ overflow: 'auto' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const j = Math.floor((e.clientX - rect.left) / cell) * strideC;
            const i = Math.floor((e.clientY - rect.top) / cell) * strideR;
            if (i >= 0 && j >= 0 && i < rows && j < cols) {
              const v = Number(W[i]?.[j]);
              setHover({ i, j, v: Number.isFinite(v) ? v : 0 });
            } else setHover(null);
          }}
          onMouseLeave={() => setHover(null)}
          style={{ display: 'block', borderRadius: 'var(--r-sm)' }}
        />
      </div>
      {Array.isArray(layer?.b) && layer.b.length > 0 && (
        <div className="wmat__bias">
          <span className="tiny muted">bias</span>
          <span className="mono tiny truncate">
            [{layer.b.map((b) => Number(b).toFixed(2)).join(', ')}]
          </span>
        </div>
      )}
    </div>
  );
}
