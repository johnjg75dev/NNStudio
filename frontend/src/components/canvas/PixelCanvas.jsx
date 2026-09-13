import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * PixelCanvas — a small grid-image editor / viewer.
 * Used by the dataset image editor and the "draw an input" test control.
 */
export default function PixelCanvas({
  width,
  height,
  values,
  channels = 1,
  zoom = 8,
  brush = 1,
  brushSize = 1,
  readOnly = false,
  showGrid = true,
  onChange,
  onStrokeEnd,
  style,
}) {
  const canvasRef = useRef(null);
  const offRef = useRef(null);
  const valuesRef = useRef(values);
  valuesRef.current = values;
  const [painting, setPainting] = useState(false);
  const dirtyRef = useRef(false);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return;
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = w * zoom;
    const ch = h * zoom;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;

    if (!offRef.current) offRef.current = document.createElement('canvas');
    const off = offRef.current;
    off.width = w;
    off.height = h;
    const octx = off.getContext('2d');
    const img = octx.createImageData(w, h);
    const data = valuesRef.current || [];
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const idx = (y * w + x) * channels;
        const r = Math.round(clamp01(data[idx] ?? 0) * 255);
        const g = channels > 1 ? Math.round(clamp01(data[idx + 1] ?? 0) * 255) : r;
        const b = channels > 2 ? Math.round(clamp01(data[idx + 2] ?? 0) * 255) : r;
        const p = (y * w + x) * 4;
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(off, 0, 0, cw, ch);

    if (showGrid && zoom >= 6) {
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 1; x < w; x += 1) {
        ctx.moveTo(x * zoom + 0.5, 0);
        ctx.lineTo(x * zoom + 0.5, ch);
      }
      for (let y = 1; y < h; y += 1) {
        ctx.moveTo(0, y * zoom + 0.5);
        ctx.lineTo(cw, y * zoom + 0.5);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(120,140,180,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, cw - 1, ch - 1);
  }, [width, height, zoom, channels, showGrid, values]);

  useEffect(() => {
    paint();
  }, [paint]);

  const pixelFromEvent = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / zoom);
    const y = Math.floor((e.clientY - rect.top) / zoom);
    return { x, y };
  };

  const applyBrush = (px, py) => {
    if (readOnly || !width || !height) return;
    const next = Array.from(valuesRef.current || []);
    const half = Math.floor(brushSize / 2);
    for (let dy = -half; dy <= half; dy += 1) {
      for (let dx = -half; dx <= half; dx += 1) {
        const x = px + dx;
        const y = py + dy;
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const idx = (y * width + x) * channels;
        if (Array.isArray(brush)) {
          for (let c = 0; c < channels; c += 1) next[idx + c] = brush[c] / 255;
        } else {
          for (let c = 0; c < channels; c += 1) next[idx + c] = brush;
        }
      }
    }
    valuesRef.current = next;
    dirtyRef.current = true;
    paint();
    onChange?.(next);
  };

  const onDown = (e) => {
    if (readOnly) return;
    e.preventDefault();
    setPainting(true);
    const { x, y } = pixelFromEvent(e);
    applyBrush(x, y);
  };
  const onMove = (e) => {
    if (!painting || readOnly) return;
    const { x, y } = pixelFromEvent(e);
    applyBrush(x, y);
  };
  const onUp = () => {
    if (!painting) return;
    setPainting(false);
    if (dirtyRef.current) {
      dirtyRef.current = false;
      onStrokeEnd?.(valuesRef.current);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
      onTouchStart={(e) => {
        const t = e.touches[0];
        onDown({ ...t, preventDefault: () => e.preventDefault(), clientX: t.clientX, clientY: t.clientY });
      }}
      onTouchMove={(e) => {
        const t = e.touches[0];
        onMove({ clientX: t.clientX, clientY: t.clientY });
      }}
      onTouchEnd={onUp}
      style={{
        display: 'block',
        imageRendering: 'pixelated',
        cursor: readOnly ? 'default' : 'crosshair',
        borderRadius: 'var(--r-sm)',
        ...style,
      }}
    />
  );
}

function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return n < 0 ? 0 : n > 1 ? 1 : n;
}
