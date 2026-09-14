import { useCallback, useEffect, useRef, useState } from 'react';
import { drawActivationCurve } from '../../lib/activations';
import { useElementSize } from '../../lib/hooks';
import { useTheme } from '../../state/ThemeContext';

/** Interactive activation-function curve with a hover readout. */
export default function ActivationChart({ type, height = 120, showDerivative = true, marker = null }) {
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const [hoverX, setHoverX] = useState(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const h = size.height || height;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawActivationCurve(ctx, size.width, h, type, theme, { hoverX, showDerivative, marker });
  }, [size.width, size.height, height, type, theme, hoverX, showDerivative, marker]);

  useEffect(() => {
    paint();
  }, [paint]);

  const onMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = rect.width / 8;
    setHoverX((e.clientX - rect.left - rect.width / 2) / scale);
  };

  return (
    <div ref={wrapRef} style={{ height, position: 'relative' }}>
      <canvas
        ref={canvasRef}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          borderRadius: 'var(--r-sm)',
          border: '1px solid var(--border-subtle)',
          cursor: 'crosshair',
        }}
      />
    </div>
  );
}
