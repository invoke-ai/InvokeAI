import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import { PluginOption, UserConfig } from 'vite';
import dts from 'vite-plugin-dts';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';

export const packageConfig: UserConfig = {
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
      external: ['react', 'react-dom', '@emotion/react'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          '@emotion/react': 'EmotionReact',
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
