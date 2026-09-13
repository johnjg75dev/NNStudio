/**
 * frontend/scripts/smoke-render.mjs
 *
 * Server-renders every page once, inside the real provider stack, and fails if
 * any of them throws.
 *
 * Why: a component can be perfectly valid JavaScript and still explode on first
 * paint — a missing import, a destructured `undefined`, a helper called before
 * the catalogue arrives. `npm run build` catches none of that, and clicking
 * through six routes by hand catches it only until the next refactor. This runs
 * in well under a second and covers all of them, including the empty-catalogue
 * state the app really is in for the first few hundred milliseconds.
 *
 * Effects never run during SSR, so canvas painting, timers and fetches are all
 * skipped — this exercises render logic only, which is exactly where the
 * crashes live.
 *
 *   node scripts/smoke-render.mjs     # from frontend/
 *   npm run smoke
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import * as esbuild from 'esbuild';

const frontendRoot = fileURLToPath(new URL('..', import.meta.url));

/** react-router warns about useLayoutEffect on every SSR render; that one is
 *  expected here, so drop it and keep every other warning visible. */
function silenceExpectedWarnings() {
  const realError = console.error;
  console.error = (...args) => {
    if (String(args[0] ?? '').includes('useLayoutEffect does nothing on the server')) return;
    realError(...args);
  };
}

// ── browser stubs ─────────────────────────────────────────────────────
// Anything a module might touch while rendering. Deliberately permissive: a
// no-op is fine, a missing global is not (that would mask the real failure).
function installBrowserStubs() {
  const noop = () => {};
  const ctx2d = new Proxy(
    { canvas: { width: 300, height: 150 } },
    {
      get: (target, prop) => (prop in target ? target[prop] : noop),
      set: () => true,
    },
  );
  const makeElement = (tagName = 'div') => ({
    tagName: String(tagName).toUpperCase(),
    style: { setProperty: noop, removeProperty: noop },
    dataset: {},
    children: [],
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    appendChild: noop,
    removeChild: noop,
    setAttribute: noop,
    getAttribute: () => null,
    addEventListener: noop,
    removeEventListener: noop,
    getBoundingClientRect: () => ({ width: 800, height: 600, top: 0, left: 0 }),
    getContext: () => ctx2d,
    width: 300,
    height: 150,
  });

  const storage = new Map();
  const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: (k) => storage.delete(k),
    clear: () => storage.clear(),
  };

  const document = {
    documentElement: makeElement('html'),
    head: makeElement('head'),
    body: makeElement('body'),
    createElement: makeElement,
    createElementNS: (_ns, tag) => makeElement(tag),
    createTextNode: (t) => ({ text: t }),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: noop,
    removeEventListener: noop,
  };

  const window = {
    document,
    localStorage,
    sessionStorage: localStorage,
    navigator: { userAgent: 'node-smoke-render' },
    location: { pathname: '/', search: '', hash: '', href: 'http://localhost/', assign: noop, replace: noop },
    history: { pushState: noop, replaceState: noop },
    devicePixelRatio: 2,
    innerWidth: 1440,
    innerHeight: 900,
    matchMedia: () => ({ matches: false, addEventListener: noop, removeEventListener: noop, addListener: noop }),
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
    addEventListener: noop,
    removeEventListener: noop,
    requestAnimationFrame: (cb) => setTimeout(() => cb(Date.now()), 0),
    cancelAnimationFrame: (id) => clearTimeout(id),
    fetch: () => new Promise(() => {}), // never resolves: SSR does not await it
    Image: class { set src(_v) {} },
    URL: globalThis.URL,
    scrollTo: noop,
  };

  // Node owns some of these as getter-only globals (`navigator` since v21), so
  // a plain assignment throws — define them instead.
  const define = (name, value) => {
    try {
      Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
    } catch {
      /* already there and not ours to replace — the stub object is enough */
    }
  };

  define('window', window);
  define('document', document);
  define('localStorage', localStorage);
  define('sessionStorage', localStorage);
  define('navigator', window.navigator);
  define('location', window.location);
  define('devicePixelRatio', 2);
  define('requestAnimationFrame', window.requestAnimationFrame);
  define('cancelAnimationFrame', window.cancelAnimationFrame);
  define('getComputedStyle', window.getComputedStyle);
  define('matchMedia', window.matchMedia);
  define('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  define('IntersectionObserver', class { observe() {} unobserve() {} disconnect() {} });
  define('MutationObserver', class { observe() {} disconnect() {} });
  define('Image', window.Image);
}

// ── generated entry ───────────────────────────────────────────────────
const ENTRY = `
import { renderToString } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';

import { ThemeProvider } from '../src/state/ThemeContext';
import { ToastProvider } from '../src/state/ToastContext';
import { CatalogProvider } from '../src/state/CatalogContext';
import { SessionProvider } from '../src/state/SessionContext';

import TrainPage from '../src/pages/TrainPage';
import DatasetsPage from '../src/pages/DatasetsPage';
import PlaygroundPage from '../src/pages/PlaygroundPage';
import ModelsPage from '../src/pages/ModelsPage';
import FunctionsPage from '../src/pages/FunctionsPage';
import LearnPage from '../src/pages/LearnPage';
import LoginPage from '../src/pages/LoginPage';
import SignupPage from '../src/pages/SignupPage';

const CASES = [
  ['/train', TrainPage],
  ['/datasets', DatasetsPage],
  ['/playground', PlaygroundPage],
  ['/models', ModelsPage],
  ['/functions', FunctionsPage],
  ['/learn', LearnPage],
  ['/login', LoginPage],
  ['/signup', SignupPage],
];

const results = [];
for (const [route, Page] of CASES) {
  try {
    const html = renderToString(
      <MemoryRouter initialEntries={[route]}>
        <ThemeProvider>
          <ToastProvider>
            <CatalogProvider>
              <SessionProvider>
                <Page />
              </SessionProvider>
            </CatalogProvider>
          </ToastProvider>
        </ThemeProvider>
      </MemoryRouter>,
    );
    results.push({ route, ok: true, bytes: html.length });
  } catch (error) {
    results.push({
      route,
      ok: false,
      error: error && error.message ? error.message : String(error),
      stack: (error && error.stack ? error.stack.split('\\n').slice(1, 5).join('\\n') : ''),
    });
  }
}

globalThis.__SMOKE_RESULTS__ = results;
`;

async function main() {
  installBrowserStubs();
  silenceExpectedWarnings();

  const work = mkdtempSync(join(tmpdir(), 'nnstudio-smoke-'));
  // The entry has to sit inside the project tree for its `../src/...` imports
  // (and node_modules resolution) to work; the bundle can live in tmp.
  const entry = join(frontendRoot, 'scripts', '.smoke-entry.jsx');
  const outfile = join(work, 'bundle.mjs');
  writeFileSync(entry, ENTRY);

  try {
    await esbuild.build({
      entryPoints: [entry],
      outfile,
      bundle: true,
      platform: 'node',
      format: 'esm',
      jsx: 'automatic',
      target: 'node18',
      logLevel: 'silent',
      absWorkingDir: frontendRoot,
      nodePaths: [join(frontendRoot, 'node_modules')],
      define: { 'process.env.NODE_ENV': '"development"' },
      // react-dom/server is CJS and calls require('stream'); esbuild's ESM
      // output needs a real require for that shim to work.
      banner: {
        js: "import { createRequire as __createRequire } from 'node:module';"
          + ' const require = __createRequire(import.meta.url);',
      },
    });

    await import(pathToFileURL(outfile).href);
    const results = globalThis.__SMOKE_RESULTS__ || [];

    const failed = results.filter((r) => !r.ok);
    for (const r of results) {
      if (r.ok) console.log(`  ✓ ${r.route.padEnd(12)} ${String(r.bytes).padStart(6)} bytes of HTML`);
      else console.log(`  ✖ ${r.route.padEnd(12)} ${r.error}\n${r.stack}\n`);
    }

    if (failed.length) {
      console.error(`\n✖ ${failed.length} of ${results.length} pages failed to render\n`);
      process.exitCode = 1;
    } else {
      console.log(`\n✓ all ${results.length} pages render`);
    }
  } finally {
    rmSync(work, { recursive: true, force: true });
    rmSync(entry, { force: true });
  }
}

main().catch((err) => {
  console.error('smoke render harness failed:', err);
  process.exitCode = 1;
});
