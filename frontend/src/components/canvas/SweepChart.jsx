import { useCallback, useEffect, useRef } from 'react';
import { drawSweepChart } from '../../lib/plot2d';
import { useElementSize } from '../../lib/hooks';
import { useTheme } from '../../state/ThemeContext';

/** Line chart for latent / node sensitivity sweeps. */
export default function SweepChart({ data, height = 130, emptyText = 'No sweep data yet' }) {
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const h = size.height || height;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawSweepChart(ctx, size.width, h, theme, data, { emptyText });
  }, [size.width, size.height, height, theme, data, emptyText]);

  useEffect(() => {
    paint();
  }, [paint]);

  return (
    <div ref={wrapRef} style={{ height, position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          borderRadius: 'var(--r-sm)',
          border: '1px solid var(--border-subtle)',
        }}
      />
    </div>
  );
}
