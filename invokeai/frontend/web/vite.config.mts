/// <reference types="vitest" />
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import type { PluginOption } from 'vite';
import { defineConfig, loadEnv } from 'vite';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import dts from 'vite-plugin-dts';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';
import { loggerContextPlugin } from './vite-plugin-logger-context';

export default defineConfig(({ mode }) => {
  // Load env file based on mode in the current working directory.
  // Set the third parameter to '' to load all env regardless of the VITE_ prefix.
  const env = loadEnv(mode, process.cwd(), '');

  // Get InvokeAI API base URL from environment variable, default to localhost
  const apiBaseUrl = env.INVOKEAI_API_BASE_URL || 'http://127.0.0.1:9090';
  const apiHost = new URL(apiBaseUrl).host;
  const wsProtocol = apiBaseUrl.startsWith('https') ? 'wss' : 'ws';

  if (mode === 'package') {
    return {
      base: './',
      plugins: [
        react(),
        eslint(),
        tsconfigPaths(),
        loggerContextPlugin(),
        visualizer(),
        dts({
          insertTypesEntry: true,
        }),
        cssInjectedByJsPlugin(),
      ],
      build: {
        /**
         * zone.js (via faro) requires max ES2015 to prevent spamming unhandled promise rejections.
         *
         * See:
         * - https://github.com/grafana/faro-web-sdk/issues/566
         * - https://github.com/angular/angular/issues/51328
         * - https://github.com/open-telemetry/opentelemetry-js/issues/3030
         */
        target: 'ES2015',
        cssCodeSplit: true,
        lib: {
          entry: path.resolve(__dirname, './src/index.ts'),
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
          app: path.resolve(__dirname, './src/app'),
          assets: path.resolve(__dirname, './src/assets'),
          common: path.resolve(__dirname, './src/common'),
          features: path.resolve(__dirname, './src/features'),
          services: path.resolve(__dirname, './src/services'),
          theme: path.resolve(__dirname, './src/theme'),
        },
      },
    };
  }

  return {
    base: './',
    plugins: [
      react(),
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
          target: `${wsProtocol}://${apiHost}`,
          ws: true,
        },
        '/openapi.json': {
          target: `${apiBaseUrl}/openapi.json`,
          rewrite: (path) => path.replace(/^\/openapi.json/, ''),
          changeOrigin: true,
        },
        '/api/': {
          target: `${apiBaseUrl}/api/`,
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
