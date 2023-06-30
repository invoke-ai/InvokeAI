import react from '@vitejs/plugin-react-swc';
import { visualizer } from 'rollup-plugin-visualizer';
import { PluginOption, UserConfig } from 'vite';
import eslint from 'vite-plugin-eslint';
import tsconfigPaths from 'vite-tsconfig-paths';

export const commonPlugins: UserConfig['plugins'] = [
  react(),
  eslint(),
  tsconfigPaths(),
  visualizer() as unknown as PluginOption,
];
