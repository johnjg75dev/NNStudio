import { useEffect, useRef } from 'react';
import { ArchDiagram } from '../../lib/archDiagrams';
import { useElementSize } from '../../lib/hooks';
import { useTheme } from '../../state/ThemeContext';

/** Educational architecture diagram (CNN, Transformer, GAN, …). */
export default function ArchDiagramCanvas({ archKey, className = '', height = null, style = null }) {
  const { theme } = useTheme();
  const [wrapRef, size] = useElementSize();
  const canvasRef = useRef(null);
  const diagramRef = useRef(null);
  if (!diagramRef.current) diagramRef.current = new ArchDiagram(theme);
  diagramRef.current.theme = theme;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 4 || size.height < 4) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    diagramRef.current.draw(ctx, size.width, size.height, archKey);
  }, [archKey, theme, size.width, size.height]);

  return (
    <div
      ref={wrapRef}
      className={`canvas-frame ${className}`}
      style={{ ...(height ? { height } : null), ...style }}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
