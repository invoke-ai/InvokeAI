import react from '@vitejs/plugin-react-swc';
import { defineConfig } from 'vite';

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
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5174,
  },
});
