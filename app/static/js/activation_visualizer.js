/**
 * static/js/activation_visualizer.js
 * Renders 2D graphs of activation functions for the Learn tab
 */

class ActivationVisualizer {
  /**
   * Draw activation function on canvas
   * @param {string} canvasId - ID of canvas element
   * @param {string} activationType - 'relu', 'leakyrelu', 'tanh', 'sigmoid', 'gelu', 'swish'
   */
  static draw(canvasId, activationType) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w;
    canvas.height = h;
    
    // Clear background
    ctx.fillStyle = "#080c10";
    ctx.fillRect(0, 0, w, h);
    
    // Draw grid and axes
    this._drawGrid(ctx, w, h);
    
    // Draw activation function
    this._drawFunction(ctx, w, h, activationType);
    
    // Draw derivative overlay (light)
    this._drawDerivative(ctx, w, h, activationType);
  }
  
  /**
   * Draw background grid and axes with tick marks and labels
   */
  static _drawGrid(ctx, w, h) {
    const cx = w / 2;
    const cy = h / 2;
    const scale = w / 8; // -4 to +4 range

    // Draw grid lines (faint)
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      const x = cx + i * scale;
      const y = cy - i * scale;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();

    // Draw tick marks and labels
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // X-axis ticks (positive and negative)
    for (let i = -4; i <= 4; i++) {
      if (i === 0) continue; // Skip origin (already covered by axes)
      const x = cx + i * scale;
      // Tick mark
      ctx.beginPath();
      ctx.moveTo(x, cy - 5);
      ctx.lineTo(x, cy + 5);
      ctx.stroke();
      // Label
      ctx.fillText(i, x, cy + 15);
    }

    // Y-axis ticks (positive and negative)
    for (let i = -4; i <= 4; i++) {
      if (i === 0) continue;
      const y = cy - i * scale;
      // Tick mark
      ctx.beginPath();
      ctx.moveTo(cx - 5, y);
      ctx.lineTo(cx + 5, y);
      ctx.stroke();
      // Label
      ctx.fillText(i, cx - 12, y);
    }

    // Origin label
    ctx.fillText("0", cx, cy + 15);
  }
  
  /**
   * Draw activation function curve
   */
static _drawFunction(ctx, w, h, activationType) {
       const cx = w / 2;
       const cy = h / 2;
       const scale = w / 8; // -4 to +4 range
       
       ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent') || "#58a6ff";
       ctx.lineWidth = 2.4;
       ctx.beginPath();
       
       let first = true;
       for (let px = 0; px < w; px += 2) {
         const x = (px - cx) / scale;
         const y = this._evaluate(x, activationType);
         const py = cy - y * scale;
         
         if (first) {
           ctx.moveTo(px, py);
           first = false;
         } else {
           ctx.lineTo(px, py);
         }
       }
       ctx.stroke();
     }
  
  /**
   * Draw derivative curve (faint)
   */
static _drawDerivative(ctx, w, h, activationType) {
     const cx = w / 2;
     const cy = h / 2;
     const scale = w / 8;
     
     ctx.strokeStyle = "rgba(100,200,255,0.3)"; // Faint blue
     ctx.lineWidth = 1.2;
     ctx.setLineDash([4, 4]);
     ctx.beginPath();
     
     let first = true;
     for (let px = 0; px < w; px += 2) {
       const x = (px - cx) / scale;
       const dy = this._derivative(x, activationType);
       const py = cy - dy * scale * 0.5; // Scale derivative smaller
       
       if (first) {
         ctx.moveTo(px, py);
         first = false;
       } else {
         ctx.lineTo(px, py);
       }
     }
     ctx.stroke();
     ctx.setLineDash([]); // Reset
   }
  
  /**
   * Evaluate activation function
   */
  static _evaluate(x, type) {
    switch (type) {
      case "relu":
        return Math.max(0, x);
      case "leakyrelu":
        return x > 0 ? x : 0.01 * x;
      case "tanh":
        return Math.tanh(x);
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "gelu":
        // Approximate GELU for display
        const t = Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x));
        return 0.5 * x * (1 + t);
      case "swish":
        const sig = 1 / (1 + Math.exp(-x));
        return x * sig;
      default:
        return 0;
    }
  }
  
  /**
   * Evaluate derivative
   */
  static _derivative(x, type) {
    const eps = 0.001;
    const f1 = this._evaluate(x + eps, type);
    const f2 = this._evaluate(x - eps, type);
    return (f1 - f2) / (2 * eps);
  }
}
