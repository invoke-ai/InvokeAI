import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const common = {
    plugins: [react()],
    build: {
      chunkSizeWarningLimit: 1500, // we don't really care about chunk size
    },
  };
  if (mode == 'development') {
    return {
      ...common,
      build: {
        ...common.build,
        sourcemap: true,
      },
    };
  } else {
    return {
      ...common,
    };
  }
});
