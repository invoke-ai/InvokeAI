/// <reference types="vitest" />
import react from '@vitejs/plugin-react-swc';
import { visualizer } from 'rollup-plugin-visualizer';
import { defineConfig } from 'vite';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';
import { loggerContextPlugin } from './vite-plugin-logger-context';

import babel from 'vite-plugin-babel';

export default defineConfig(({ mode }) => {
  return {
    base: './',
    plugins: [
      react(),
      babel({
        babelConfig: {
          plugins: ['babel-plugin-react-compiler'],
        },
      }),
      mode !== 'test' && eslint({ failOnError: mode === 'production', failOnWarning: mode === 'production' }),
      tsconfigPaths(),
      mode !== 'test' && loggerContextPlugin(),
      visualizer(),
    ],
    build: {
      chunkSizeWarningLimit: 1500,
    },
    server: {
      proxy: {
        '/ws/socket.io': {
          target: 'ws://127.0.0.1:9090',
          ws: true,
        },
        '/openapi.json': {
          target: 'http://127.0.0.1:9090/openapi.json',
          rewrite: (path) => path.replace(/^\/openapi.json/, ''),
          changeOrigin: true,
        },
        '/api/': {
          target: 'http://127.0.0.1:9090/api/',
          rewrite: (path) => path.replace(/^\/api/, ''),
          changeOrigin: true,
        },
      },
      host: '0.0.0.0',
    },
    test: {
      typecheck: {
        enabled: true,
        ignoreSourceErrors: true,
      },
      coverage: {
        provider: 'v8',
        all: false,
        reporter: ['html'],
      },
    },
  };
});
