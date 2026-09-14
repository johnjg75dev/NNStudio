/**
 * lib/archDiagrams.js — educational diagrams for non-trainable architectures.
 *
 * Drawn in a fixed 800×420 design space and scaled to fit the canvas, so the
 * diagrams stay legible at any panel size.
 */
import { palette, withAlpha } from './colors';

const DW = 800;
const DH = 420;
const SANS = "'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, monospace";

export const DIAGRAM_KEYS = [
  'mlp',
  'autoencoder',
  'cnn',
  'transformer',
  'vit',
  'vae',
  'diffusion',
  'gan',
  'rnn',
];

export class ArchDiagram {
  constructor(theme = 'dark') {
    this.theme = theme;
  }

  draw(ctx, w, h, key) {
    const pal = palette(this.theme);
    this.ctx = ctx;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = pal.canvasBg;
    ctx.fillRect(0, 0, w, h);

    // dot grid backdrop
    ctx.fillStyle = pal.grid;
    for (let x = 24; x < w; x += 24) {
      for (let y = 24; y < h; y += 24) ctx.fillRect(x, y, 1, 1);
    }

    const s = Math.min(w / DW, h / DH);
    ctx.save();
    ctx.translate(w / 2 - (DW / 2) * s, h / 2 - (DH / 2) * s);
    ctx.scale(s, s);
    this.pal = pal;

    const fn = {
      cnn: () => this.cnn(),
      transformer: () => this.transformer(),
      vit: () => this.vit(),
      vae: () => this.vae(),
      diffusion: () => this.diffusion(),
      gan: () => this.gan(),
      rnn: () => this.rnn(),
      autoencoder: () => this.autoencoder(),
      mlp: () => this.mlp(),
    }[key];

    if (fn) fn();
    else this.generic(key);
    ctx.restore();
  }

  // ── primitives ──
  box(x, y, w, h, label, sub, color, opts = {}) {
    const ctx = this.ctx;
    const pal = this.pal;
    ctx.save();
    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, withAlpha(color, opts.solid ? 0.42 : 0.24));
    grad.addColorStop(1, withAlpha(color, opts.solid ? 0.2 : 0.08));
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 7);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = withAlpha(color, 0.75);
    ctx.lineWidth = 1.4;
    ctx.stroke();

    const lines = String(label).split('\n');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.font = `700 11.5px ${SANS}`;
    const textBlock = lines.length * 13 + (sub ? String(sub).split('\n').length * 10 + 2 : 0);
    let ty = y + h / 2 - textBlock / 2 + 7;
    lines.forEach((line) => {
      ctx.fillText(line, x + w / 2, ty);
      ty += 13;
    });
    if (sub) {
      ctx.fillStyle = pal.text3;
      ctx.font = `500 9px ${MONO}`;
      ty += 1;
      String(sub).split('\n').forEach((line) => {
        ctx.fillText(line, x + w / 2, ty);
        ty += 10;
      });
    }
    ctx.restore();
  }

  arrow(x1, y1, x2, y2, color) {
    const ctx = this.ctx;
    const c = color || withAlpha(this.pal.text3, 0.7);
    ctx.save();
    ctx.strokeStyle = c;
    ctx.fillStyle = c;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(a - 0.32), y2 - 8 * Math.sin(a - 0.32));
    ctx.lineTo(x2 - 8 * Math.cos(a + 0.32), y2 - 8 * Math.sin(a + 0.32));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  note(text, x = DW / 2, y = DH - 16) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = this.pal.text3;
    ctx.font = `500 11px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  caption(text, x, y, color) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = color || this.pal.text3;
    ctx.font = `600 9.5px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  chain(blocks, cy = DH / 2, gap = 20) {
    const total = blocks.reduce((s, b) => s + b.w + gap, 0) - gap;
    let x = DW / 2 - total / 2;
    blocks.forEach((b, i) => {
      const bh = b.h || 56;
      this.box(x, cy - bh / 2, b.w, bh, b.l, b.s, b.c);
      if (i < blocks.length - 1) this.arrow(x + b.w + 3, cy, x + b.w + gap - 3, cy);
      x += b.w + gap;
    });
  }

  set ctx(value) {
    this._ctx = value;
  }

  get ctx() {
    return this._ctx;
  }

  // ── diagrams ──
  mlp() {
    const pal = this.pal;
    const cols = [3, 5, 5, 2];
    const xs = [180, 320, 460, 600];
    const cy = DH / 2 - 10;
    const positions = cols.map((n, ci) =>
      Array.from({ length: n }, (_, i) => ({
        x: xs[ci],
        y: cy - ((n - 1) * 34) / 2 + i * 34,
      })),
    );
    for (let l = 0; l < positions.length - 1; l += 1) {
      positions[l].forEach((a) => {
        positions[l + 1].forEach((b) => {
          this.ctx.strokeStyle = withAlpha(pal.accent, 0.16);
          this.ctx.lineWidth = 1;
          this.ctx.beginPath();
          this.ctx.moveTo(a.x, a.y);
          this.ctx.lineTo(b.x, b.y);
          this.ctx.stroke();
        });
      });
    }
    positions.forEach((layer, l) => {
      layer.forEach((p, i) => {
        const t = (i + 1) / (layer.length + 1);
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, 11, 0, Math.PI * 2);
        const g = this.ctx.createLinearGradient(p.x - 11, p.y - 11, p.x + 11, p.y + 11);
        g.addColorStop(0, withAlpha(l === 0 ? pal.text3 : l === 3 ? pal.pos : pal.accent, 0.55 + t * 0.4));
        g.addColorStop(1, withAlpha(l === 0 ? pal.text3 : l === 3 ? pal.pos : pal.accent, 0.18));
        this.ctx.fillStyle = g;
        this.ctx.fill();
        this.ctx.strokeStyle = withAlpha(pal.text1, 0.35);
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
      });
    });
    ['Input', 'Hidden 1', 'Hidden 2', 'Output'].forEach((label, i) =>
      this.caption(label.toUpperCase(), xs[i], cy - 108, pal.text3),
    );
    this.note('MLP — every neuron connects to every neuron in the next layer. Universal approximator.');
  }

  autoencoder() {
    const pal = this.pal;
    const widths = [120, 80, 34, 80, 120];
    const labels = ['Input x', 'Encoder', 'Latent z', 'Decoder', 'Output x̂'];
    let x = 90;
    const cy = DH / 2 - 16;
    widths.forEach((w, i) => {
      const h = 150 - i * 18;
      const hh = i > 2 ? 150 - (4 - i) * 18 : h;
      const color = [pal.accent, pal.violet, pal.warn, pal.violet, pal.pos][i];
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.roundRect(x, cy - hh / 2, w, hh, 6);
      const g = this.ctx.createLinearGradient(x, cy - hh / 2, x + w, cy + hh / 2);
      g.addColorStop(0, withAlpha(color, 0.3));
      g.addColorStop(1, withAlpha(color, 0.08));
      this.ctx.fillStyle = g;
      this.ctx.fill();
      this.ctx.strokeStyle = withAlpha(color, 0.7);
      this.ctx.lineWidth = 1.3;
      this.ctx.stroke();
      // inner "neuron" rows
      const rows = Math.max(2, Math.round(hh / 22));
      for (let r = 0; r < rows; r += 1) {
        const ry = cy - hh / 2 + 12 + r * ((hh - 24) / Math.max(1, rows - 1));
        this.ctx.beginPath();
        this.ctx.arc(x + w / 2, ry, 3.2, 0, Math.PI * 2);
        this.ctx.fillStyle = withAlpha(color, 0.85);
        this.ctx.fill();
      }
      this.ctx.restore();
      this.caption(labels[i], x + w / 2, cy + hh / 2 + 18, pal.text2);
      if (i < widths.length - 1) {
        this.arrow(x + w + 6, cy, x + w + 34, cy, withAlpha(pal.text3, 0.6));
      }
      x += w + 40;
    });
    this.note('Autoencoder — compress to a bottleneck, then reconstruct. Learns a compact latent code.');
  }

  cnn() {
    const pal = this.pal;
    this.chain(
      [
        { l: 'Input', s: 'H×W×3', c: pal.accent, w: 62, h: 62 },
        { l: 'Conv2D', s: '32 filters', c: pal.pos, w: 72, h: 68 },
        { l: 'MaxPool', s: '÷2', c: pal.teal, w: 62, h: 58 },
        { l: 'Conv2D', s: '64 filters', c: pal.pos, w: 72, h: 68 },
        { l: 'MaxPool', s: '÷2', c: pal.teal, w: 62, h: 58 },
        { l: 'Flatten\n+ Dense', s: '512', c: pal.warn, w: 74, h: 62 },
        { l: 'Softmax', s: 'N classes', c: pal.neg, w: 74, h: 56 },
      ],
      DH / 2 - 20,
      18,
    );
    this.note('CNN — sliding kernels detect local features; pooling builds translation invariance.');
  }

  transformer() {
    const pal = this.pal;
    const lyrs = [
      { y: DH - 74, l: 'Input Tokens', s: 'word ids', c: pal.accent, w: 200, h: 32 },
      { y: DH - 128, l: 'Embed + Pos. Encoding', s: 'd_model', c: pal.accent, w: 250, h: 32 },
      { y: DH - 196, l: 'Multi-Head Attention', s: 'softmax(QKᵀ/√d)·V', c: pal.pos, w: 262, h: 38 },
      { y: DH - 250, l: 'Add & LayerNorm', s: 'residual', c: pal.warn, w: 240, h: 28 },
      { y: DH - 300, l: 'Feed-Forward', s: 'Linear→ReLU→Linear', c: pal.violet, w: 240, h: 36 },
      { y: DH - 344, l: 'Add & LayerNorm', s: 'residual', c: pal.warn, w: 240, h: 28 },
      { y: 26, l: 'Output / Task Head', s: 'classify or next-token', c: pal.neg, w: 200, h: 32 },
    ];
    lyrs.forEach((l) => this.box(DW / 2 - l.w / 2, l.y, l.w, l.h, l.l, l.s, l.c));
    for (let i = 0; i < lyrs.length - 1; i += 1) {
      this.arrow(DW / 2, lyrs[i].y - 2, DW / 2, lyrs[i + 1].y + lyrs[i + 1].h + 2, withAlpha(pal.text3, 0.65));
    }
    this.ctx.save();
    this.ctx.strokeStyle = withAlpha(pal.text3, 0.5);
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([4, 3]);
    this.ctx.strokeRect(DW / 2 - 148, DH - 360, 296, 190);
    this.ctx.restore();
    this.caption('× N layers', DW / 2 + 172, DH - 264, pal.text3);
    this.note('Transformer — “Attention Is All You Need” (Vaswani et al. 2017).');
  }

  vit() {
    const pal = this.pal;
    const ctx = this.ctx;
    const imgS = 116;
    const imgX = 40;
    const imgY = DH / 2 - imgS / 2 - 14;
    ctx.save();
    ctx.strokeStyle = withAlpha(pal.text3, 0.5);
    ctx.lineWidth = 1;
    ctx.strokeRect(imgX, imgY, imgS, imgS);
    const ps = imgS / 4;
    for (let r = 0; r < 4; r += 1) {
      for (let c = 0; c < 4; c += 1) {
        ctx.fillStyle = withAlpha(pal.accent, 0.12 + ((r * 4 + c) % 7) * 0.07);
        ctx.fillRect(imgX + c * ps + 1, imgY + r * ps + 1, ps - 2, ps - 2);
      }
    }
    ctx.restore();
    this.caption('Image → 4×4 patches', imgX + imgS / 2, imgY + imgS + 16);
    this.arrow(imgX + imgS + 10, DH / 2 - 14, imgX + imgS + 52, DH / 2 - 14);
    this.box(imgX + imgS + 56, DH / 2 - 66, 86, 104, 'Patch\nEmbed', 'Linear + CLS', pal.pos);
    this.arrow(imgX + imgS + 146, DH / 2 - 14, imgX + imgS + 188, DH / 2 - 14);
    this.box(imgX + imgS + 192, DH / 2 - 72, 118, 116, 'Transformer\nEncoder', '×L self-attn\n+ FFN', pal.violet);
    this.arrow(imgX + imgS + 314, DH / 2 - 14, imgX + imgS + 356, DH / 2 - 14);
    this.box(imgX + imgS + 360, DH / 2 - 42, 96, 60, 'MLP Head', '[CLS]→classes', pal.neg);
    this.note('ViT — “An Image is Worth 16×16 Words” (Dosovitskiy et al. 2020).');
  }

  vae() {
    const pal = this.pal;
    this.chain(
      [
        { l: 'Input x', s: 'data', c: pal.accent, w: 76 },
        { l: 'Encoder', s: 'q(z|x)', c: pal.pos, w: 86 },
        { l: 'μ, σ²', s: 'latent', c: pal.warn, w: 80 },
        { l: 'z = μ + σ·ε', s: 'reparameterise', c: pal.violet, w: 96 },
        { l: 'Decoder', s: 'p(x|z)', c: pal.teal, w: 86 },
        { l: 'Output x̂', s: 'reconstruction', c: pal.neg, w: 90 },
      ],
      DH / 2 - 22,
      22,
    );
    this.note('VAE — Loss = reconstruction error + KL(q(z|x) ‖ p(z))  ·  Kingma & Welling 2013');
  }

  diffusion() {
    const pal = this.pal;
    const ctx = this.ctx;
    const cy = DH / 2 - 14;
    ctx.save();
    ctx.fillStyle = pal.text2;
    ctx.font = `700 11px ${SANS}`;
    ctx.textAlign = 'left';
    ctx.fillText('Forward process — add noise', 46, cy - 84);
    ctx.restore();

    const fc = [pal.pos, pal.accent, pal.warn, pal.violet, pal.neg];
    for (let i = 0; i < 5; i += 1) {
      const x = 60 + i * 96;
      const r = 24 - i * 2.4;
      ctx.beginPath();
      ctx.arc(x, cy - 46, r, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(fc[i], 0.55);
      ctx.fill();
      this.caption(i === 0 ? 'x₀ data' : i === 4 ? 'xᴛ noise' : `x${i}`, x, cy - 12, pal.text3);
      if (i < 4) this.arrow(x + r + 6, cy - 46, x + 96 - (24 - (i + 1) * 2.4) - 6, cy - 46, withAlpha(pal.text3, 0.5));
    }

    ctx.save();
    ctx.fillStyle = pal.text2;
    ctx.font = `700 11px ${SANS}`;
    ctx.textAlign = 'left';
    ctx.fillText('Reverse process — U-Net denoises', 46, cy + 40);
    ctx.restore();

    const rc = [pal.neg, pal.violet, pal.warn, pal.accent, pal.pos];
    for (let i = 0; i < 5; i += 1) {
      const x = 60 + i * 96;
      const r = 12 + i * 2.6;
      ctx.beginPath();
      ctx.arc(x, cy + 84, r, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(rc[i], 0.2);
      ctx.strokeStyle = withAlpha(rc[i], 0.85);
      ctx.lineWidth = 1.4;
      ctx.fill();
      ctx.stroke();
      this.caption(i === 0 ? 'xᴛ' : i === 4 ? 'x₀' : `x${4 - i}`, x, cy + 116, pal.text3);
      if (i < 4) this.arrow(x + r + 6, cy + 84, x + 96 - (12 + (i + 1) * 2.6) - 6, cy + 84, withAlpha(pal.violet, 0.7));
    }
    this.box(DW - 226, cy + 20, 190, 74, 'U-Net  εθ(xₜ, t)', 'predicts the noise', pal.violet);
    this.note('Diffusion — DDPM (Ho et al. 2020). Stable Diffusion adds VAE latents + CLIP text conditioning.');
  }

  gan() {
    const pal = this.pal;
    const cy = DH / 2 - 26;
    this.box(56, cy - 48, 122, 96, 'Generator\nG(z)', 'noise → fake data', pal.pos);
    this.arrow(182, cy, 236, cy);
    this.box(240, cy - 56, 130, 112, 'Discriminator\nD(x)', 'real or fake?', pal.neg);
    this.arrow(374, cy - 24, 428, cy - 24);
    this.box(432, cy - 44, 96, 40, 'Real → 1', 'P(real)', pal.accent);
    this.arrow(374, cy + 24, 428, cy + 24);
    this.box(432, cy + 4, 96, 40, 'Fake → 0', 'P(fake)', pal.warn);
    this.box(56, cy - 108, 76, 30, 'Noise z', 'random', pal.violet);
    this.arrow(94, cy - 76, 108, cy - 50);
    this.box(152, cy - 108, 100, 30, 'Real data', 'training set', pal.accent);
    this.arrow(212, cy - 76, 280, cy - 58);
    this.note('GAN — G: log(1−D(G(z)))  ·  D: log D(x) + log(1−D(G(z)))  ·  Goodfellow et al. 2014');
  }

  rnn() {
    const pal = this.pal;
    const cells = 5;
    const cw = 86;
    const ch = 56;
    const gap = 34;
    const cy = DH / 2 - 40;
    const sx = DW / 2 - (cells * (cw + gap) - gap) / 2;
    for (let t = 0; t < cells; t += 1) {
      const x = sx + t * (cw + gap);
      this.box(x, cy - ch / 2, cw, ch, t === 2 ? 'LSTM Cell' : 'RNN Cell', `h_${t}`, pal.violet);
      this.arrow(x + cw / 2, cy + ch / 2 + 4, x + cw / 2, cy + ch / 2 + 30, withAlpha(pal.text3, 0.55));
      this.caption(`x_${t}`, x + cw / 2, cy + ch / 2 + 44, pal.text3);
      this.arrow(x + cw / 2, cy - ch / 2 - 4, x + cw / 2, cy - ch / 2 - 28, withAlpha(pal.text3, 0.55));
      this.caption(`y_${t}`, x + cw / 2, cy - ch / 2 - 34, pal.text3);
      if (t < cells - 1) this.arrow(x + cw + 3, cy, x + cw + gap - 3, cy, withAlpha(pal.pos, 0.8));
    }
    const gx = DW / 2 - 178;
    const gy = DH - 96;
    [
      ['Forget', 'f = σ(Wf·[h,x]+b)', pal.neg],
      ['Input', 'i = σ(Wi·[h,x]+b)', pal.pos],
      ['Output', 'o = σ(Wo·[h,x]+b)', pal.accent],
    ].forEach(([n, f, c], i) => this.box(gx + i * 122, gy, 114, 34, n, f, c));
    this.caption('LSTM GATES', DW / 2, gy - 10, pal.text3);
    this.note('RNN / LSTM — sequential memory via a hidden state  ·  Hochreiter & Schmidhuber 1997');
  }

  generic(key) {
    const pal = this.pal;
    this.box(DW / 2 - 150, DH / 2 - 40, 300, 70, key.toUpperCase(), 'no diagram for this architecture', pal.text3);
    this.note('See the Learn page for concept notes.');
  }
}

/** Convenience: draw a diagram in one call. */
export function drawArchDiagram(ctx, w, h, key, theme = 'dark') {
  const diagram = new ArchDiagram(theme);
  diagram.ctx = ctx;
  diagram.draw(ctx, w, h, key);
}
