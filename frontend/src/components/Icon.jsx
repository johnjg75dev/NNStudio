import { memo } from 'react';

/**
 * Icon — a tiny inline SVG icon set (24×24, stroke-based).
 * Replaces the emoji sprinkled through the legacy templates.
 */
const PATHS = {
  play: <path d="M6 4.5v15l13-7.5-13-7.5z" fill="currentColor" stroke="none" />,
  pause: (
    <>
      <rect x="6" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" />
      <rect x="14" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" />
    </>
  ),
  stop: <rect x="5.5" y="5.5" width="13" height="13" rx="2" fill="currentColor" stroke="none" />,
  step: (
    <>
      <path d="M5 4.5v15l11-7.5L5 4.5z" fill="currentColor" stroke="none" />
      <path d="M19 5v14" />
    </>
  ),
  reset: (
    <>
      <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
      <path d="M3 3v5h5" />
    </>
  ),
  build: (
    <>
      <path d="M12 2.8 21 7.4v9.2L12 21.2 3 16.6V7.4l9-4.6z" />
      <path d="M3 7.4 12 12l9-4.6M12 12v9.2" />
    </>
  ),
  layers: (
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5z" />
      <path d="m3 13 9 5 9-5" />
    </>
  ),
  network: (
    <>
      <circle cx="5" cy="6" r="2.2" />
      <circle cx="5" cy="18" r="2.2" />
      <circle cx="19" cy="12" r="2.2" />
      <path d="M7 7.1 17 11M7 16.9 17 13" />
    </>
  ),
  sliders: (
    <>
      <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" />
      <circle cx="16" cy="6" r="2" />
      <circle cx="10" cy="12" r="2" />
      <circle cx="18" cy="18" r="2" />
    </>
  ),
  gauge: (
    <>
      <path d="M12 21a9 9 0 1 0-9-9 9 9 0 0 0 9 9z" />
      <path d="m12 12 4-3" />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
    </>
  ),
  chart: (
    <>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="m7 15 3.5-4.5 3 2.5L20 7" />
    </>
  ),
  activity: <path d="M3 12h3.5L9 5l3.5 14L15.5 12H21" />,
  database: (
    <>
      <ellipse cx="12" cy="6" rx="8" ry="3.2" />
      <path d="M4 6v12c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2V6" />
      <path d="M4 12c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2" />
    </>
  ),
  beaker: (
    <>
      <path d="M9 3h6M10 3v6.2L5.2 17A2.6 2.6 0 0 0 7.5 21h9a2.6 2.6 0 0 0 2.3-4L14 9.2V3" />
      <path d="M7.4 14.5h9.2" />
    </>
  ),
  code: (
    <>
      <path d="m8.5 8-4.5 4 4.5 4M15.5 8l4.5 4-4.5 4M13.5 5l-3 14" />
    </>
  ),
  book: (
    <>
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z" />
      <path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20v3H6.5" />
    </>
  ),
  save: (
    <>
      <path d="M5 3h11l3 3v15H5z" />
      <path d="M8 3v6h7V3M8 21v-6h8v6" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4" />
      <path d="m7.5 8.5 4.5-4.5 4.5 4.5" />
      <path d="M4 16v3.5A1.5 1.5 0 0 0 5.5 21h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </>
  ),
  download: (
    <>
      <path d="M12 4v12" />
      <path d="m7.5 11.5 4.5 4.5 4.5-4.5" />
      <path d="M4 16v3.5A1.5 1.5 0 0 0 5.5 21h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="3.6" />
      <path d="M4.8 20.4a7.4 7.4 0 0 1 14.4 0" />
    </>
  ),
  logout: (
    <>
      <path d="M14 4h4a1.6 1.6 0 0 1 1.6 1.6v12.8A1.6 1.6 0 0 1 18 20h-4" />
      <path d="M10 8l-4 4 4 4M6 12h9" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4" />
    </>
  ),
  moon: <path d="M20 14.4A8.4 8.4 0 0 1 9.6 4a8.5 8.5 0 1 0 10.4 10.4z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  trash: (
    <>
      <path d="M4 7h16M9.5 7V4.6h5V7M6.5 7l1 13h9l1-13" />
      <path d="M10.5 11v5.5M13.5 11v5.5" />
    </>
  ),
  close: <path d="m6 6 12 12M18 6 6 18" />,
  check: <path d="m5 12.5 4.5 4.5L19 7" />,
  chevronLeft: <path d="m14.5 5.5-6.5 6.5 6.5 6.5" />,
  chevronRight: <path d="m9.5 5.5 6.5 6.5-6.5 6.5" />,
  chevronDown: <path d="m5.5 9.5 6.5 6.5 6.5-6.5" />,
  chevronUp: <path d="m5.5 14.5 6.5-6.5 6.5 6.5" />,
  arrowUp: <path d="M12 20V5M6 11l6-6 6 6" />,
  arrowDown: <path d="M12 4v15M6 13l6 6 6-6" />,
  arrowRight: <path d="M4 12h15M13 6l6 6-6 6" />,
  search: (
    <>
      <circle cx="11" cy="11" r="6.4" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7.5 19.4l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 3 14.6H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 7.5l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.6 1.6 0 0 0 10 3.1V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0 1.1 2.7H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1.3z" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12S6 5.8 12 5.8 21.5 12 21.5 12 18 18.2 12 18.2 2.5 12 2.5 12z" />
      <circle cx="12" cy="12" r="2.9" />
    </>
  ),
  eyeOff: (
    <>
      <path d="M4 4l16 16" />
      <path d="M9.6 5.9A9.6 9.6 0 0 1 12 5.8c6 0 9.5 6.2 9.5 6.2a17 17 0 0 1-3.2 4M6.4 8.1A16.7 16.7 0 0 0 2.5 12S6 18.2 12 18.2a9.7 9.7 0 0 0 3.6-.7" />
      <path d="M9.9 10.1a3 3 0 0 0 4.1 4.2" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="8.4" />
      <circle cx="12" cy="12" r="4.4" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  history: (
    <>
      <path d="M3.5 12a8.5 8.5 0 1 0 2.8-6.3L3.5 8.4" />
      <path d="M3.5 3.6v4.8h4.8" />
      <path d="M12 7.8V12l3 1.8" />
    </>
  ),
  sparkles: (
    <>
      <path d="m12 3.5 1.7 4.6 4.6 1.7-4.6 1.7L12 16.1l-1.7-4.6L5.7 9.8l4.6-1.7L12 3.5z" />
      <path d="m18.5 15.5.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2z" />
    </>
  ),
  wand: (
    <>
      <path d="m4 20 10-10M14.5 4.2l.9 2.3 2.3.9-2.3.9-.9 2.3-.9-2.3-2.3-.9 2.3-.9.9-2.3z" />
      <path d="m19.5 13.5.6 1.5 1.5.6-1.5.6-.6 1.5-.6-1.5-1.5-.6 1.5-.6.6-1.5z" />
    </>
  ),
  keyboard: (
    <>
      <rect x="2.5" y="6" width="19" height="12" rx="2" />
      <path d="M6 9.5h.01M9.5 9.5h.01M13 9.5h.01M16.5 9.5h.01M6 12.8h.01M9.5 12.8h.01M13 12.8h.01M16.5 12.8h.01M8 15.6h8" />
    </>
  ),
  grid: (
    <>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.4" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.4" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.4" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1.4" />
    </>
  ),
  table: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M3.5 9.5h17M9.5 9.5v10M3.5 14.5h17" />
    </>
  ),
  image: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <circle cx="8.8" cy="9.6" r="1.6" />
      <path d="m4.5 17 4.8-4.4 3.4 3 2.8-2.4 4 3.8" />
    </>
  ),
  file: (
    <>
      <path d="M6 3h8l4 4v14H6z" />
      <path d="M14 3v4h4" />
    </>
  ),
  folder: <path d="M3.5 6.5A1.5 1.5 0 0 1 5 5h4l2 2.5h8a1.5 1.5 0 0 1 1.5 1.5v9A1.5 1.5 0 0 1 19 19.5H5A1.5 1.5 0 0 1 3.5 18z" />,
  copy: (
    <>
      <rect x="8.5" y="8.5" width="12" height="12" rx="2" />
      <path d="M15.5 5.5A2 2 0 0 0 13.5 3.5h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2" />
    </>
  ),
  grip: (
    <>
      <circle cx="9" cy="6" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="15" cy="6" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="9" cy="12" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="15" cy="12" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="9" cy="18" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="15" cy="18" r="1.4" fill="currentColor" stroke="none" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="8.6" />
      <path d="M12 11v5.4M12 7.9h.01" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3.6 21.4 20H2.6z" />
      <path d="M12 9.6v4.2M12 16.8h.01" />
    </>
  ),
  error: (
    <>
      <circle cx="12" cy="12" r="8.6" />
      <path d="m9 9 6 6M15 9l-6 6" />
    </>
  ),
  zoomIn: (
    <>
      <circle cx="11" cy="11" r="6.4" />
      <path d="m16 16 4.5 4.5M11 8.6v4.8M8.6 11h4.8" />
    </>
  ),
  zoomOut: (
    <>
      <circle cx="11" cy="11" r="6.4" />
      <path d="m16 16 4.5 4.5M8.6 11h4.8" />
    </>
  ),
  fit: (
    <>
      <path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15" />
    </>
  ),
  maximize: (
    <>
      <path d="M4 9V4h5M20 15v5h-5M15 4h5v5M9 20H4v-5" />
    </>
  ),
  minimize: <path d="M9 4v5H4M15 20v-5h5M20 9h-5V4M4 15h5v5" />,
  panelRight: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M14.5 4.5v15" />
    </>
  ),
  cpu: (
    <>
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
      <rect x="9.8" y="9.8" width="4.4" height="4.4" rx="1" />
      <path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21" />
    </>
  ),
  brain: (
    <>
      <path d="M12 5.2a2.9 2.9 0 0 0-5.5 1.2A2.8 2.8 0 0 0 4.6 9a2.9 2.9 0 0 0 .8 4A2.9 2.9 0 0 0 7 18.2a2.9 2.9 0 0 0 5 .8z" />
      <path d="M12 5.2a2.9 2.9 0 0 1 5.5 1.2A2.8 2.8 0 0 1 19.4 9a2.9 2.9 0 0 1-.8 4 2.9 2.9 0 0 1-1.6 5.2 2.9 2.9 0 0 1-5 .8z" />
    </>
  ),
  bolt: <path d="M13.5 2.5 5 13.5h5.5L10 21.5 19 10.5h-5.6z" />,
  cube: (
    <>
      <path d="m12 2.8 8.4 4.6v9.2L12 21.2 3.6 16.6V7.4z" />
      <path d="M3.6 7.4 12 12l8.4-4.6M12 12v9.2" />
    </>
  ),
  list: <path d="M8 6.5h12M8 12h12M8 17.5h12M4 6.5h.01M4 12h.01M4 17.5h.01" />,
  shuffle: (
    <>
      <path d="M3 6.5h3.6l9.8 11H21M3 17.5h3.6l3.2-3.6M14.6 9.2l2.8-2.7H21" />
      <path d="m18.4 3.6 2.6 2.9-2.6 2.9M18.4 14.6l2.6 2.9-2.6 2.9" />
    </>
  ),
  dice: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="3" />
      <circle cx="8.6" cy="8.6" r="1.3" fill="currentColor" stroke="none" />
      <circle cx="15.4" cy="15.4" r="1.3" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" />
    </>
  ),
  wave: <path d="M2.5 12c2.2-6 4-6 6.2 0s4 6 6.2 0 4-6 6.6 0" />,
  cursor: (
    <>
      <path d="M5.5 3.5 18 11l-5.4 1.6L10.4 18 5.5 3.5z" />
    </>
  ),
  focus: (
    <>
      <path d="M3.5 8.5v-5h5M20.5 8.5v-5h-5M3.5 15.5v5h5M20.5 15.5v5h-5" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  sweep: (
    <>
      <path d="M3.5 17.5h17" />
      <path d="M6 17.5V13M10 17.5V8.5M14 17.5v-7M18 17.5V5" />
    </>
  ),
  link: (
    <>
      <path d="M10.5 13.5a3.6 3.6 0 0 0 5.2.2l2.6-2.6a3.7 3.7 0 0 0-5.2-5.2l-1.4 1.4" />
      <path d="M13.5 10.5a3.6 3.6 0 0 0-5.2-.2l-2.6 2.6a3.7 3.7 0 0 0 5.2 5.2l1.4-1.4" />
    </>
  ),
  filter: <path d="M3.5 5.5h17l-6.6 7.7v5.6l-3.8 2v-7.6z" />,
  logo: (
    <>
      <path d="M12 2.6 20.4 7v10L12 21.4 3.6 17V7z" />
      <circle cx="12" cy="12" r="2.6" />
      <path d="M12 9.4V6.2M12 14.6v3.2M9.7 10.7 6.9 9.1M14.3 13.3l2.8 1.6M9.7 13.3l-2.8 1.6M14.3 10.7l2.8-1.6" />
    </>
  ),
};

const Icon = memo(function Icon({ name, size = 16, className, strokeWidth = 1.7, ...rest }) {
  const path = PATHS[name];
  if (!path) return null;
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className}
      {...rest}
    >
      {path}
    </svg>
  );
});

export default Icon;
