/** Seven-segment display glyph driven by 7 network outputs. */
export function SevenSegment({ values = [], size = 46, showLabel = false, label = '' }) {
  const on = values.map((v) => Number(v) > 0.5);
  const segments = [
    { x1: 4, y1: 3, x2: 22, y2: 3 },
    { x1: 23, y1: 4, x2: 23, y2: 20 },
    { x1: 23, y1: 22, x2: 23, y2: 38 },
    { x1: 4, y1: 39, x2: 22, y2: 39 },
    { x1: 3, y1: 22, x2: 3, y2: 38 },
    { x1: 3, y1: 4, x2: 3, y2: 20 },
    { x1: 4, y1: 21, x2: 22, y2: 21 },
  ];
  return (
    <figure className="seg7" style={{ width: size * 0.66 }}>
      <svg viewBox="0 0 26 42" width={size * 0.66} height={size} aria-hidden="true">
        {segments.map((s, i) => (
          <line
            key={i}
            x1={s.x1}
            y1={s.y1}
            x2={s.x2}
            y2={s.y2}
            stroke={on[i] ? 'var(--pos)' : 'var(--surface-strong)'}
            strokeWidth="3"
            strokeLinecap="round"
            style={{
              filter: on[i] ? 'drop-shadow(0 0 4px color-mix(in srgb, var(--pos) 70%, transparent))' : 'none',
              transition: 'stroke 120ms ease',
            }}
          />
        ))}
      </svg>
      {showLabel && <figcaption className="mono tiny">{label}</figcaption>}
    </figure>
  );
}

/** Grid of seven-segment digits, used for the seg7 task. */
export function SevenSegmentGrid({ samples, pick = 'pred', onSelect, labelFor }) {
  return (
    <div className="seg-grid">
      {samples.map((s, i) => (
        <button
          key={i}
          type="button"
          className="seg-cell"
          onClick={() => onSelect?.(s)}
          title={labelFor?.(s) || `Sample ${i + 1}`}
        >
          <SevenSegment values={s[pick] || []} size={40} />
          {labelFor && <span className="mono xs muted">{labelFor(s)}</span>}
        </button>
      ))}
    </div>
  );
}

/** Convert a 4-bit input vector to its hex digit (seg7 task). */
export function fourBitToHex(x = []) {
  const digit = ((x[0] > 0.5 ? 1 : 0) << 3) | ((x[1] > 0.5 ? 1 : 0) << 2) | ((x[2] > 0.5 ? 1 : 0) << 1) | (x[3] > 0.5 ? 1 : 0);
  return digit < 10 ? String(digit) : String.fromCharCode(65 + digit - 10);
}

export default SevenSegment;
