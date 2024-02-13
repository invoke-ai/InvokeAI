/// <reference types="vitest" />
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import type { PluginOption } from 'vite';
import { defineConfig } from 'vite';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import dts from 'vite-plugin-dts';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(({ mode }) => {
  if (mode === 'package') {
    return {
      base: './',
      plugins: [
        react(),
        eslint(),
        tsconfigPaths(),
        visualizer() as unknown as PluginOption,
        dts({
          insertTypesEntry: true,
        }),
        cssInjectedByJsPlugin(),
      ],
      build: {
        cssCodeSplit: true,
        lib: {
          entry: path.resolve(__dirname, '../src/index.ts'),
          name: 'InvokeAIUI',
          fileName: (format) => `invoke-ai-ui.${format}.js`,
        },
        rollupOptions: {
          external: ['react', 'react-dom', '@emotion/react', '@chakra-ui/react', '@invoke-ai/ui-library'],
          output: {
            globals: {
              react: 'React',
              'react-dom': 'ReactDOM',
              '@emotion/react': 'EmotionReact',
              '@invoke-ai/ui-library': 'UiLibrary',
            },
          },
        },
      },
      resolve: {
        alias: {
          app: path.resolve(__dirname, '../src/app'),
          assets: path.resolve(__dirname, '../src/assets'),
          common: path.resolve(__dirname, '../src/common'),
          features: path.resolve(__dirname, '../src/features'),
          services: path.resolve(__dirname, '../src/services'),
          theme: path.resolve(__dirname, '../src/theme'),
        },
      },
    };
  }

  return {
    base: './',
    plugins: [react(), mode !== 'test' && eslint(), tsconfigPaths(), visualizer() as unknown as PluginOption],
    build: {
      chunkSizeWarningLimit: 1500,
    },
    server: {
      // Proxy HTTP requests to the flask server
      proxy: {
        // Proxy socket.io to the nodes socketio server
        '/ws/socket.io': {
          target: 'ws://127.0.0.1:9090',
          ws: true,
        },
        // Proxy openapi schema definiton
        '/openapi.json': {
          target: 'http://127.0.0.1:9090/openapi.json',
          rewrite: (path) => path.replace(/^\/openapi.json/, ''),
          changeOrigin: true,
        },
        // proxy nodes api
        '/api/v1': {
          target: 'http://127.0.0.1:9090/api/v1',
          rewrite: (path) => path.replace(/^\/api\/v1/, ''),
          changeOrigin: true,
        },
      },
    },
    test: {
      //
    },
  };
});
