/**
 * static/js/plot2d.js
 * 2D Plot Renderer for decision boundaries and function plots
 */

class Plot2DRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.width = this.canvas.offsetWidth;
    this.height = this.canvas.offsetHeight;
    
    // Set canvas internal resolution
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    
    this.resolution = 20; // Grid resolution for decision boundary
  }

  /**
   * Render 2D decision boundary or function plot
   * @param {Object} snapshot - Network snapshot with topology, layers, func
   * @param {Array} samples - Training data samples
   * @param {Object} options - Render options {showBoundary, showPoints}
   */
  draw(snapshot, samples, options = {}) {
    if (!this.canvas || !this.ctx) return;
    
    // Update dimensions if canvas was resized
    this.width = this.canvas.offsetWidth;
    this.height = this.canvas.offsetHeight;
    
    // Guard: don't draw if canvas hasn't been laid out yet
    if (this.width <= 0 || this.height <= 0) return;
    
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    
    const { showBoundary = true, showPoints = true } = options;
    
    if (!snapshot || !this.ctx) return;
    
    const fn = snapshot.func || {};
    const inputs = fn.inputs ?? 0;
    
    // Clear canvas
    this.ctx.fillStyle = "#080c10";
    this.ctx.fillRect(0, 0, this.width, this.height);
    
    // Draw based on input dimension
    if (inputs === 2) {
      this._draw2DDecisionBoundary(snapshot, samples, showBoundary, showPoints);
    } else if (inputs === 1) {
      this._draw1DFunctionPlot(snapshot, samples, showBoundary, showPoints);
    } else {
      this._drawPlaceholder("2D plot only for 1-2 input functions");
    }
  }

  /**
   * Draw decision boundary for 2-input classification
   */
  _draw2DDecisionBoundary(snapshot, samples, showBoundary, showPoints) {
    const fn = snapshot.func || {};
    const w = this.width;
    const h = this.height;
    const res = this.resolution;
    const cellW = w / res;
    const cellH = h / res;

    // Draw prediction heatmap
    if (showBoundary) {
      const imageData = this.ctx.createImageData(w, h);
      const data = imageData.data;

      for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
          // Normalize pixel to [0,1] range
          const x = px / w;
          const y = py / h;

          // Predict at this point
          const pred = this._predictAt([x, y], snapshot);
          const conf = Math.abs(pred - 0.5) * 2; // 0-1 confidence
          
          // Color based on prediction
          const idx = (py * w + px) * 4;
          if (pred > 0.5) {
            // Class 1: Blue-ish
            data[idx]     = 30 + conf * 80;
            data[idx + 1] = 100 + conf * 50;
            data[idx + 2] = 180 + conf * 50;
            data[idx + 3] = 200;
          } else {
            // Class 0: Red-ish
            data[idx]     = 180 + conf * 50;
            data[idx + 1] = 50 + conf * 30;
            data[idx + 2] = 50 + conf * 30;
            data[idx + 3] = 200;
          }
        }
      }
      this.ctx.putImageData(imageData, 0, 0);
    }

    // Draw data points
    if (showPoints && samples && samples.length > 0) {
      samples.forEach(sample => {
        const x = sample.x[0] * w;
        const y = sample.x[1] * h;
        const label = sample.y[0];

        // Draw point
        this.ctx.fillStyle = label > 0.5 ? "#58a6ff" : "#f85149";
        this.ctx.beginPath();
        this.ctx.arc(x, y, 4, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw border
        this.ctx.strokeStyle = "#ffffff";
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
      });
    }

    // Draw border
    this.ctx.strokeStyle = "#30363d";
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(0, 0, w, h);
  }

  /**
   * Draw function plot for 1-input regression
   */
  _draw1DFunctionPlot(snapshot, samples, showFunc, showPoints) {
    const fn = snapshot.func || {};
    const w = this.width;
    const h = this.height;
    const margin = 20;
    const plotW = w - 2 * margin;
    const plotH = h - 2 * margin;

    // Draw axes
    this.ctx.strokeStyle = "#30363d";
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(margin, h - margin);
    this.ctx.lineTo(w - margin, h - margin);
    this.ctx.moveTo(margin, margin);
    this.ctx.lineTo(margin, h - margin);
    this.ctx.stroke();

    // Draw function curve (predicted)
    if (showFunc) {
      this.ctx.strokeStyle = "#58a6ff";
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      let first = true;
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const pred = this._predictAt([x], snapshot);
        const px = margin + (x * plotW);
        const py = (h - margin) - (pred * plotH);
        
        if (first) {
          this.ctx.moveTo(px, py);
          first = false;
        } else {
          this.ctx.lineTo(px, py);
        }
      }
      this.ctx.stroke();
    }

    // Draw data points and actual values
    if (showPoints && samples && samples.length > 0) {
      samples.forEach(sample => {
        const x = sample.x[0];
        const y = sample.y[0];
        const px = margin + (x * plotW);
        const py = (h - margin) - (y * plotH);

        // Draw point
        this.ctx.fillStyle = "#3fb950";
        this.ctx.beginPath();
        this.ctx.arc(px, py, 3, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw prediction point if showFunc
        if (showFunc) {
          const pred = this._predictAt([x], snapshot);
          const predPy = (h - margin) - (pred * plotH);
          this.ctx.fillStyle = "#58a6ff";
          this.ctx.beginPath();
          this.ctx.arc(px, predPy, 2, 0, Math.PI * 2);
          this.ctx.fill();
        }
      });
    }

    // Draw border
    this.ctx.strokeStyle = "#30363d";
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(margin - 1, margin - 1, plotW + 2, plotH + 2);

    // Draw axis labels
    this.ctx.fillStyle = "#8b949e";
    this.ctx.font = "10px sans-serif";
    this.ctx.textAlign = "center";
    this.ctx.fillText("0", margin, h - margin + 12);
    this.ctx.fillText("1", w - margin, h - margin + 12);
    this.ctx.textAlign = "right";
    this.ctx.fillText("1", margin - 4, margin + 4);
    this.ctx.fillText("0", margin - 4, h - margin + 4);
  }

  /**
   * Predict network output for given input
   */
  _predictAt(input, snapshot) {
    if (!snapshot || !snapshot.built) return 0;

    // Simple forward pass through layers
    let activation = input;
    const layers = snapshot.layers || [];

    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const W = layer.W || [];
      const b = layer.b || [];
      const type = (layer.type || "").toLowerCase();

      // Matrix multiply: a = W @ x + b
      let nextActivation = new Array(W.length).fill(0);
      if (Array.isArray(W[0])) {
        for (let i = 0; i < W.length; i++) {
          if (Array.isArray(W[i])) {
            for (let j = 0; j < W[i].length; j++) {
              nextActivation[i] += W[i][j] * (activation[j] ?? 0);
            }
          }
          nextActivation[i] += b[i] ?? 0;
        }
      }

      // Apply activation
      const act = snapshot.activation || "tanh";
      activation = nextActivation.map(x => this._activate(x, act));
    }

    return activation[0] ?? 0;
  }

  /**
   * Apply activation function
   */
  _activate(x, type = "tanh") {
    const t = (type || "").toLowerCase();
    switch (t) {
      case "relu":
        return Math.max(0, x);
      case "leakyrelu":
        return x > 0 ? x : 0.01 * x;
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "tanh":
        return Math.tanh(x);
      case "linear":
        return x;
      default:
        return Math.tanh(x);
    }
  }

  _drawPlaceholder(text) {
    this.ctx.fillStyle = "#6e7681";
    this.ctx.font = "12px sans-serif";
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.fillText(text, this.width / 2, this.height / 2);
  }
}
