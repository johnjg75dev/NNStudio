import { arrayToGrid } from '../../lib/grid';

/**
 * PixelPreview — a tiny read-only rendering of a flat image-shaped vector.
 * Used for image outputs in the I/O browser and the playground.
 */
export function PixelPreview({ data, shape, small = false, max = 96 }) {
  const grid = arrayToGrid(data, shape);
  if (!grid.w || !grid.h) return <span className="tiny muted">—</span>;
  const cell = small
    ? Math.max(2, Math.min(9, Math.floor(max / Math.max(grid.w, grid.h))))
    : Math.max(3, Math.min(14, Math.floor((max * 1.6) / Math.max(grid.w, grid.h))));
  const values = grid.values;
  const peak = Math.max(0.001, ...values.map((v) => Math.abs(v)));
  return (
    <div
      className="pixel-mini"
      style={{ width: grid.w * cell, height: grid.h * cell }}
      title={`${grid.w}×${grid.h}${grid.c > 1 ? `×${grid.c}` : ''}`}
    >
      {values.map((v, i) => {
        const t = Math.max(0, Math.min(1, v / peak));
        return (
          <span
            key={i}
            style={{
              left: (i % grid.w) * cell,
              top: Math.floor(i / grid.w) * cell,
              width: cell > 3 ? cell - 1 : cell,
              height: cell > 3 ? cell - 1 : cell,
              background: `color-mix(in srgb, var(--accent) ${Math.round(t * 100)}%, var(--surface-strong))`,
            }}
          />
        );
      })}
    </div>
  );
}

export default PixelPreview;
