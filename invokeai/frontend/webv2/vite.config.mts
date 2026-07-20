import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

// Override with e.g. INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 when the
// backend dev server runs on a non-default port.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');

export default defineConfig({
  base: './',
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
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

          if (id.includes('/node_modules/react-icons/')) {
            return 'react-icons';
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
