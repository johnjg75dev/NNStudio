/**
 * static/js/trainer.js
 * TrainingController — owns the requestAnimationFrame training loop.
 * Fires events that the UI listens to; never touches the DOM directly.
 *
 * Events emitted via EventTarget:
 *   "step"     — after each batch of training steps  (detail: metrics + snapshot)
 *   "stopped"  — when training is paused/stopped
 *   "error"    — on API failure (detail: { message })
 */
class TrainingController extends EventTarget {
  constructor() {
    super();
    this._running   = false;
    this._rafId     = null;
    this._steps     = 10;      // steps per frame
    this._lr        = 0.01;
    this._snapshot  = null;    // last received snapshot
  }

  get running()   { return this._running; }
  get snapshot()  { return this._snapshot; }

  configure({ steps, lr }) {
    if (steps !== undefined) this._steps = Math.max(1, steps);
    if (lr    !== undefined) this._lr    = lr;
  }

  start() {
    if (this._running) return;
    this._running = true;
    this._loop();
  }

  stop() {
    this._running = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this.dispatchEvent(new Event("stopped"));
  }

  toggle() {
    this._running ? this.stop() : this.start();
  }

  // Force one step even when stopped (manual advance)
  async stepOnce() {
    await this._tick();
  }

  // ── internal ──
  _loop() {
    if (!this._running) return;
    this._tick().then(() => {
      if (this._running) {
        this._rafId = requestAnimationFrame(() => this._loop());
      }
    });
  }

  async _tick() {
    try {
      const data = await API.trainStep(this._steps, this._lr);
      this._snapshot = data;
      this.dispatchEvent(new CustomEvent("step", { detail: data }));
    } catch (e) {
      this.stop();
      this.dispatchEvent(new CustomEvent("error", { detail: { message: e.message } }));
    }
  }
}
