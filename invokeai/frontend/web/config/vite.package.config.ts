import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import { PluginOption, UserConfig } from 'vite';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';
import dts from 'vite-plugin-dts';

export const packageConfig: UserConfig = {
  plugins: [
    react(),
    eslint(),
    tsconfigPaths(),
    visualizer() as unknown as PluginOption,
    dts({
      insertTypesEntry: true,
    }),
  ],
  build: {
    chunkSizeWarningLimit: 1500,
    lib: {
      entry: path.resolve(__dirname, '../src/index.ts'),
      name: 'InvokeAIUI',
      fileName: (format) => `invoke-ai-ui.${format}.js`,
    },
    rollupOptions: {
      external: ['react', 'react-dom'],
      output: {
        globals: {
          react: 'React',
        },
      },
    },
  },
};
