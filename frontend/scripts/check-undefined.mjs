/**
 * frontend/scripts/check-undefined.mjs
 *
 * Fails on identifiers that are used but never declared or imported.
 *
 * Vite/Rollup treat an unknown name as a global and bundle it without a word,
 * so a missing import does not break the build — it becomes a runtime
 * `ReferenceError` deep inside a component (this is how `useIoShape` took down
 * the whole training studio). Babel's scope analysis finds them statically.
 *
 *   node scripts/check-undefined.mjs        # from frontend/
 *   npm run check                           # same thing
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Babel ships as a transitive dependency of @vitejs/plugin-react. This script
// runs as a `prebuild` hook, so if it is ever missing we skip the check with a
// warning rather than making the build itself impossible.
let parse;
let traverse;
try {
  ({ parse } = await import('@babel/parser'));
  const traversePkg = await import('@babel/traverse');
  traverse = traversePkg.default?.default ?? traversePkg.default;
} catch {
  console.warn('⚠ @babel/parser unavailable — skipping the undefined-reference check');
  process.exit(0);
}

const here = fileURLToPath(new URL('..', import.meta.url)); // frontend/

/** Real globals: language built-ins plus the browser APIs this app touches. */
const GLOBALS = new Set(
  [
    // language
    'undefined', 'NaN', 'Infinity', 'globalThis', 'arguments',
    'Object', 'Function', 'Boolean', 'Symbol', 'Number', 'BigInt', 'Math',
    'Date', 'String', 'RegExp', 'Array', 'JSON', 'Map', 'Set', 'WeakMap',
    'WeakSet', 'Promise', 'Proxy', 'Reflect', 'Intl', 'Error', 'TypeError',
    'RangeError', 'SyntaxError', 'ReferenceError', 'EvalError', 'URIError',
    'AggregateError', 'ArrayBuffer', 'SharedArrayBuffer', 'DataView',
    'Float32Array', 'Float64Array', 'Int8Array', 'Int16Array', 'Int32Array',
    'Uint8Array', 'Uint8ClampedArray', 'Uint16Array', 'Uint32Array',
    'BigInt64Array', 'BigUint64Array', 'Atomics', 'WebAssembly',
    'isNaN', 'isFinite', 'parseInt', 'parseFloat', 'encodeURI',
    'encodeURIComponent', 'decodeURI', 'decodeURIComponent', 'escape',
    'unescape', 'eval', 'structuredClone', 'queueMicrotask',
    // browser
    'window', 'self', 'document', 'navigator', 'location', 'history',
    'screen', 'devicePixelRatio', 'localStorage', 'sessionStorage',
    'console', 'performance', 'crypto',
    'fetch', 'URL', 'URLSearchParams', 'Blob', 'File', 'FileList',
    'FileReader', 'FormData', 'Headers', 'Request', 'Response',
    'AbortController', 'AbortSignal', 'TextEncoder', 'TextDecoder',
    'Image', 'Audio', 'Canvas', 'OffscreenCanvas', 'Path2D',
    'ImageBitmap', 'ImageData', 'DOMParser', 'XMLHttpRequest',
    'Event', 'CustomEvent', 'MouseEvent', 'PointerEvent', 'KeyboardEvent',
    'TouchEvent', 'WheelEvent', 'InputEvent', 'FocusEvent', 'DragEvent',
    'MessageEvent', 'ErrorEvent', 'ResizeObserver', 'IntersectionObserver',
    'MutationObserver', 'Worker', 'SharedWorker', 'BroadcastChannel',
    'EventSource', 'WebSocket', 'MessagePort', 'MessageChannel',
    'Node', 'Element', 'HTMLElement', 'HTMLCanvasElement',
    'HTMLInputElement', 'HTMLTextAreaElement', 'DocumentFragment',
    'getComputedStyle', 'matchMedia', 'requestAnimationFrame',
    'cancelAnimationFrame', 'requestIdleCallback', 'cancelIdleCallback',
    'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval',
    'alert', 'confirm', 'prompt', 'print', 'atob', 'btoa', 'scrollTo',
    'scrollBy', 'find', 'focus', 'blur', 'open', 'close', 'postMessage',
    // node (build scripts / SSR-ish edge cases)
    'process', 'global', 'Buffer', 'module', 'exports', 'require',
    '__dirname', '__filename',
  ],
);

function* sourceFiles(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) yield* sourceFiles(full);
    else if (/\.(js|jsx|mjs)$/.test(entry)) yield full;
  }
}

const roots = [join(here, 'src')];
if (process.argv[2]) roots.length = 0, roots.push(resolve(process.argv[2]));

const findings = [];

for (const root of roots) {
  for (const file of sourceFiles(root)) {
    const code = readFileSync(file, 'utf8');
    let ast;
    try {
      ast = parse(code, { sourceType: 'module', plugins: ['jsx'] });
    } catch (err) {
      findings.push({ file, line: err.loc?.line ?? 0, name: `<parse error: ${err.message}>` });
      continue;
    }

    const reported = new Set();
    const report = (name, node) => {
      const key = `${name}:${node?.loc?.start?.line}`;
      if (reported.has(key)) return;
      reported.add(key);
      findings.push({ file, line: node?.loc?.start?.line ?? 0, name });
    };

    traverse(ast, {
      ReferencedIdentifier(path) {
        const { name } = path.node;
        if (!name || GLOBALS.has(name)) return;
        // `noGlobals = true`: only real bindings count, so built-ins fall
        // through to the GLOBALS check above rather than being silently bound.
        // NB: no `scope.hasGlobal()` here — Babel answers true for *any* free
        // variable, which is precisely the case we want to report. The GLOBALS
        // whitelist above is the only thing allowed to excuse an unbound name.
        if (path.scope.hasBinding(name, true)) return;
        report(name, path.node);
      },
      // Component tags: <Foo /> — babel marks these as referenced too, but be
      // explicit so a missing component import can never slip past.
      JSXOpeningElement(path) {
        const nameNode = path.node.name;
        const name = nameNode?.name;
        if (typeof name !== 'string' || GLOBALS.has(name)) return;
        // Lowercase tags are intrinsic elements (<div>, <path>, <circle>);
        // only component tags resolve to a binding.
        if (/^[a-z]/.test(name)) return;
        if (path.scope.hasBinding(name, true)) return;
        report(name, nameNode);
      },
    });
  }
}

if (findings.length) {
  console.error(`\n✖ ${findings.length} undefined reference(s):\n`);
  for (const f of findings) {
    console.error(`  ${relative(here, f.file)}:${f.line}  ${f.name}`);
  }
  console.error('\nEach of these is a runtime ReferenceError waiting to happen — add the import.\n');
  process.exit(1);
}

console.log('✓ no undefined references');
