import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

import babel from '@rolldown/plugin-babel';

// Override with e.g. INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 when the
// backend dev server runs on a non-default port.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');

export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('/node_modules/')) {
            return undefined;
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
