import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

import { chunkSourceManifest } from './scripts/chunk-source-manifest.mjs';

// Override with e.g. INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 when the
// backend dev server runs on a non-default port.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');
const PROJECT_ROOT = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  base: './',
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Tiny first-party modules imported by both the eager shell and lazy
          // feature chunks (store selectors, the palette's open state, the
          // date-token parser); one shared chunk instead of a file per module.
          if (
            id.endsWith('/platform/state/selectors.ts') ||
            id.endsWith('/workbench/palette/paletteStore.ts') ||
            id.endsWith('/platform/search/dateTokens.ts') ||
            id.endsWith('/platform/performance/semanticReady.ts')
          ) {
            return 'shared';
          }

          // Mixed-gallery item helpers cross the eager editor shell and its
          // lazy Gallery/Preview chunks. Keep them with the already-eager
          // gallery state projection instead of emitting one extra request.
          if (
            id.endsWith('/features/gallery/core/items.ts') ||
            id.endsWith('/features/gallery/ui/galleryStateView.ts')
          ) {
            return 'gallery-state';
          }

          if (!id.includes('/node_modules/')) {
            return undefined;
          }

          // ag-psd (+ its only dependents pako/base64-js) is loaded LAZILY via a
          // dynamic `import('ag-psd')` at PSD-export time. Keep it in its own chunk
          // so it never lands in the eagerly-preloaded `vendor` bundle — otherwise
          // the manualChunks catch-all below would pull it into the initial load.
          if (
            id.includes('/node_modules/ag-psd/') ||
            id.includes('/node_modules/pako/') ||
            id.includes('/node_modules/base64-js/')
          ) {
            return 'ag-psd';
          }

          // Same story as ag-psd: `yaml` is reached only through a dynamic
          // `import('yaml')` in wildcard import/export, and the catch-all below
          // would otherwise make a parser nobody has asked for part of the
          // initial load.
          if (id.includes('/node_modules/yaml/')) {
            return 'yaml';
          }

          if (id.includes('/node_modules/react-icons/')) {
            return 'react-icons';
          }

          // The Workflow editor's graph canvas. XYFlow and the d3 modules it
          // pulls in are reachable only from the Workflow widget, which is
          // lazy — but the `vendor` catch-all below made them launchpad-eager.
          if (id.includes('/node_modules/@xyflow/') || /\/node_modules\/d3-[^/]+\//.test(id)) {
            return 'workflow-vendor';
          }

          // Editor-only pointer and drag interaction libraries: perfect-freehand
          // belongs to Canvas stroke geometry, dnd-kit to widget and tab
          // reordering. Neither is reachable from the launchpad.
          if (id.includes('/node_modules/perfect-freehand/') || id.includes('/node_modules/@dnd-kit/')) {
            return 'editor-interactions';
          }

          if (id.includes('/node_modules/@chakra-ui/') || id.includes('/node_modules/@emotion/')) {
            return 'chakra';
          }

          return 'vendor';
        },
      },
    },
  },
  plugins: [
    react(),
    babel({
      presets: [reactCompilerPreset()],
    }),
    chunkSourceManifest({ projectRoot: PROJECT_ROOT }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@app': fileURLToPath(new URL('./src/app', import.meta.url)),
      '@assets': fileURLToPath(new URL('./src/assets', import.meta.url)),
      '@features': fileURLToPath(new URL('./src/features', import.meta.url)),
      '@platform': fileURLToPath(new URL('./src/platform', import.meta.url)),
      '@theme': fileURLToPath(new URL('./src/platform/ui/theme', import.meta.url)),
      '@workbench': fileURLToPath(new URL('./src/workbench', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    proxy: {
      '/api/': {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        target: `${BACKEND_URL}/api/`,
      },
      '/openapi.json': {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/openapi.json/, ''),
        target: `${BACKEND_URL}/openapi.json`,
      },
      '/ws/socket.io': {
        target: BACKEND_WS_URL,
        ws: true,
      },
    },
  },
});
