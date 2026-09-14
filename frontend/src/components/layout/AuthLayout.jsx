import { useEffect, useRef } from 'react';
import Icon from '../Icon';
import { useTheme } from '../../state/ThemeContext';

/** Split-screen layout for /login and /signup with a live network backdrop. */
export default function AuthLayout({ title, children }) {
  return (
    <div className="auth">
      <aside className="auth__aside">
        <DecorNetwork />
        <div className="auth__aside-content">
          <div className="auth__brand">
            <span className="rail__brand" style={{ marginBottom: 0 }}>
              <Icon name="logo" size={20} />
            </span>
            <div>
              <b>NNStudio</b>
              <span className="tiny">neural network studio</span>
            </div>
          </div>
          <h2>
            Build it. Train it.
            <br />
            Watch it think.
          </h2>
          <p>
            A pure-NumPy training engine with a live network graph, decision boundaries, latent-space
            probing and an architecture library — all in the browser.
          </p>
          <ul className="auth__points">
            {[
              ['network', 'Every weight, drawn live as it learns'],
              ['beaker', 'Custom tasks in Python or JavaScript'],
              ['chart', 'Loss curves, boundaries and sweeps'],
              ['layers', '12 layer types, 5 optimizers, 6 activations'],
            ].map(([icon, text]) => (
              <li key={text}>
                <Icon name={icon} size={15} />
                <span>{text}</span>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <main className="auth__main">
        <div className="auth__card">
          <h1>{title}</h1>
          {children}
        </div>
        <p className="auth__foot tiny muted">
          Sessions live in memory on the server — your trained network stays put while you explore.
        </p>
      </main>
    </div>
  );
}

/** Decorative animated graph: nodes drift, edges pulse. Purely aesthetic. */
function DecorNetwork() {
  const { theme } = useTheme();
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    let raf = 0;
    let w = 0;
    let h = 0;

    const layers = [4, 6, 6, 3];
    const nodes = [];
    const build = () => {
      nodes.length = 0;
      const padX = w * 0.16;
      const padY = h * 0.16;
      layers.forEach((n, l) => {
        for (let i = 0; i < n; i += 1) {
          nodes.push({
            l,
            i,
            x: padX + (l / (layers.length - 1)) * (w - padX * 2),
            y: padY + (n === 1 ? h / 2 : (i / (n - 1)) * (h - padY * 2)),
            phase: Math.random() * Math.PI * 2,
          });
        }
      });
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.parentElement.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      build();
    };

    const edges = [];
    const buildEdges = () => {
      edges.length = 0;
      for (let l = 0; l < layers.length - 1; l += 1) {
        const from = nodes.filter((n) => n.l === l);
        const to = nodes.filter((n) => n.l === l + 1);
        from.forEach((a) => to.forEach((b) => edges.push([a, b, Math.random()])));
      }
    };

    const draw = (t) => {
      ctx.clearRect(0, 0, w, h);
      const accent = theme === 'dark' ? [110, 160, 255] : [47, 111, 228];
      const violet = theme === 'dark' ? [169, 123, 255] : [139, 92, 246];
      edges.forEach(([a, b, seed]) => {
        const pulse = 0.5 + 0.5 * Math.sin(t / 900 + seed * 8);
        ctx.strokeStyle = `rgba(${accent.join(',')},${0.05 + pulse * 0.14})`;
        ctx.lineWidth = 0.7 + pulse * 0.8;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });
      nodes.forEach((n) => {
        const pulse = 0.5 + 0.5 * Math.sin(t / 700 + n.phase);
        const c = n.l % 2 ? violet : accent;
        const r = 3 + pulse * 2.4;
        const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
        g.addColorStop(0, `rgba(${c.join(',')},${0.25 + pulse * 0.3})`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c.join(',')},${0.55 + pulse * 0.45})`;
        ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    };

    resize();
    buildEdges();
    raf = requestAnimationFrame(draw);
    const ro = new ResizeObserver(() => {
      resize();
      buildEdges();
    });
    ro.observe(canvas.parentElement);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [theme]);

  return <canvas ref={ref} className="auth__decor" aria-hidden="true" />;
}
