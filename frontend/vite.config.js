import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Flask app is the single origin in production: `npm run build` emits into
// frontend/dist and app/api/page_routes.py serves it (API calls are same-origin).
// During development the Vite server proxies every backend route to Flask so
// cookies (Flask-Login / session id) keep working.
const FLASK = process.env.FLASK_ORIGIN || 'http://127.0.0.1:5000';

const proxied = [
  '/api',
  '/login',
  '/signup',
  '/logout',
  '/check-username',
  '/admin',
];

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: false,
    proxy: Object.fromEntries(
      proxied.map((path) => [path, { target: FLASK, changeOrigin: true }]),
    ),
  },
});
