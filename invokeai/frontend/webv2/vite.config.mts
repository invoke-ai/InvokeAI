import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

import babel from '@rolldown/plugin-babel';

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
        target: 'http://127.0.0.1:9090/api/',
      },
      '/openapi.json': {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/openapi.json/, ''),
        target: 'http://127.0.0.1:9090/openapi.json',
      },
      '/ws/socket.io': {
        target: 'ws://127.0.0.1:9090',
        ws: true,
      },
    },
  },
});
