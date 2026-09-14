import { Modal } from '../ui';

const GROUPS = [
  {
    title: 'Training',
    items: [
      { keys: ['Space'], label: 'Start / pause training' },
      { keys: ['B'], label: 'Rebuild the network' },
      { keys: ['R'], label: 'Re-initialise weights' },
    ],
  },
  {
    title: 'Navigation',
    items: [
      { keys: ['1'], label: 'Training studio' },
      { keys: ['2'], label: 'Datasets' },
      { keys: ['3'], label: 'Playground' },
      { keys: ['4'], label: 'Model library' },
      { keys: ['5'], label: 'Custom functions' },
      { keys: ['6'], label: 'Learn' },
    ],
  },
  {
    title: 'Canvas',
    items: [
      { keys: ['Scroll'], label: 'Zoom the network graph' },
      { keys: ['Drag'], label: 'Pan the network graph' },
      { keys: ['Click'], label: 'Inspect a neuron' },
    ],
  },
  {
    title: 'General',
    items: [
      { keys: ['T'], label: 'Toggle light / dark theme' },
      { keys: ['?'], label: 'Show this dialog' },
      { keys: ['Esc'], label: 'Close a dialog' },
    ],
  },
];

export default function ShortcutsDialog({ open, onClose }) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Keyboard shortcuts"
      subtitle="Everything in the studio is reachable without the mouse."
      icon="keyboard"
      size="wide"
    >
      <div className="shortcuts">
        {GROUPS.map((group) => (
          <section key={group.title}>
            <h4 className="section-title">{group.title}</h4>
            <ul>
              {group.items.map((item) => (
                <li key={item.label}>
                  <span>{item.label}</span>
                  <span className="row" style={{ gap: 4 }}>
                    {item.keys.map((k) => (
                      <kbd key={k}>{k}</kbd>
                    ))}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </Modal>
  );
}
