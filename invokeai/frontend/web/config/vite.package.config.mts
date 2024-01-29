import path from 'path';
import type { UserConfig } from 'vite';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import dts from 'vite-plugin-dts';

import { commonPlugins } from './common.mjs';

export const packageConfig: UserConfig = {
  base: './',
  plugins: [
    ...commonPlugins,
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
