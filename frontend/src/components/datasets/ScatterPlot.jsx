import { useMemo, useState } from 'react';
import { palette } from '../../lib/colors';
import { useTheme } from '../../state/ThemeContext';

const CLASS_COLORS = ['#6ea0ff', '#3ddc84', '#ffc861', '#f0883e', '#a97bff', '#2dd4bf', '#ff7b9c', '#c0ca33'];

/**
 * ScatterPlot — 2-input datasets drawn as an SVG scatter, coloured by class.
 * Hovering a point reports its coordinates and target.
 */
export default function ScatterPlot({ samples = [], height = 260, onSelect }) {
  const { theme } = useTheme();
  const pal = palette(theme);
  const [hover, setHover] = useState(null);

  const { points, xDom, yDom, classes } = useMemo(() => {
    const pts = [];
    const xs = [];
    const ys = [];
    const cls = new Set();
    samples.forEach((s, i) => {
      const x = Number(s.x?.[0]);
      const y = Number(s.x?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      xs.push(x);
      ys.push(y);
      const c = classOf(s.y);
      cls.add(c);
      pts.push({ i, x, y, c, sample: s });
    });
    const dom = (arr) => {
      if (!arr.length) return [0, 1];
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      const pad = (max - min || 1) * 0.08;
      return [min - pad, max + pad];
    };
    return { points: pts, xDom: dom(xs), yDom: dom(ys), classes: [...cls].sort() };
  }, [samples]);

  if (points.length < 2) {
    return <p className="tiny muted">A scatter needs at least two samples with two numeric inputs.</p>;
  }

  const W = 100; // viewBox units; the SVG scales to the container
  const H = 100;
  const sx = (v) => ((v - xDom[0]) / (xDom[1] - xDom[0] || 1)) * W;
  const sy = (v) => H - ((v - yDom[0]) / (yDom[1] - yDom[0] || 1)) * H;

  return (
    <div className="scatter">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height }} role="img" aria-label="Sample scatter plot">
        <rect x="0" y="0" width={W} height={H} fill={pal.canvasBg} />
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={t} stroke={pal.grid} strokeWidth="0.2">
            <line x1={t * W} y1="0" x2={t * W} y2={H} />
            <line x1="0" y1={t * H} x2={W} y2={t * H} />
          </g>
        ))}
        {points.map((p) => (
          <circle
            key={p.i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={hover === p.i ? 2.4 : 1.6}
            fill={CLASS_COLORS[p.c % CLASS_COLORS.length]}
            fillOpacity={hover === null || hover === p.i ? 0.9 : 0.45}
            stroke={pal.canvasBg}
            strokeWidth="0.25"
            onMouseEnter={() => setHover(p.i)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onSelect?.(p.sample)}
            style={{ cursor: onSelect ? 'pointer' : 'default' }}
          />
        ))}
      </svg>
      <div className="scatter__legend">
        {classes.map((c) => (
          <span key={c} className="legend-item">
            <span className="legend-dot" style={{ background: CLASS_COLORS[c % CLASS_COLORS.length] }} />
            class {c}
          </span>
        ))}
        <span className="legend-item mono tiny" style={{ marginLeft: 'auto' }}>
          x {xDom[0].toFixed(2)}…{xDom[1].toFixed(2)} · y {yDom[0].toFixed(2)}…{yDom[1].toFixed(2)}
        </span>
      </div>
      {hover !== null && (
        <div className="scatter__readout mono tiny">
          #{hover} · x={points.find((p) => p.i === hover)?.x.toFixed(3)} y=
          {points.find((p) => p.i === hover)?.y.toFixed(3)} · class{' '}
          {points.find((p) => p.i === hover)?.c}
        </div>
      )}
    </div>
  );
}

/** Collapse a target vector into a single class index. */
function classOf(y) {
  if (!Array.isArray(y) || !y.length) return 0;
  if (y.length === 1) return Number(y[0]) > 0.5 ? 1 : 0;
  let best = 0;
  let bestV = -Infinity;
  y.forEach((v, i) => {
    const n = Number(v) || 0;
    if (n > bestV) {
      bestV = n;
      best = i;
    }
  });
  return best;
}
