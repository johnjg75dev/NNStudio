## 2026-03-29 - [Spacebar Hotkey Conflict in Code Editors]
**Learning:** Global hotkey listeners (e.g., Space for Train/Pause) interfere with text entry in `TEXTAREA` and `INPUT` elements. Accessibility and UX require explicit focus checks before triggering global actions.
**Action:** Always check `event.target.tagName` or `event.target.isContentEditable` in global keydown listeners to avoid blocking native browser behaviors in interactive fields.

## 2026-03-29 - [Categorized Task Library for Complex Selectors]
**Learning:** As the number of training tasks (Math, Logic, Vision, Custom) grows, a standard `<select>` dropdown becomes overwhelming and difficult to navigate. A categorized modal with visual cards improves discoverability and provides space for detailed descriptions and I/O metadata.
**Action:** Use a "Task Library" modal with filters when the selection list exceeds ~15 items or requires rich metadata display.

## 2026-03-29 - [Visual Parity in Network Rendering]
**Learning:** When a backend model contains utility layers (Dropout, BatchNorm, Flatten) that don't have traditional "neurons," omitting them from the frontend visualization creates a mental model mismatch for the user.
**Action:** Represent utility layers with distinct visual styles (dashed borders, unique badges) to maintain 1:1 parity with the actual network topology while distinguishing them from computational layers.
