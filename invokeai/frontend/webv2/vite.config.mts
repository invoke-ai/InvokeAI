import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

import { chunkSourceManifest } from './scripts/chunk-source-manifest.mjs';

// Override with e.g. INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 when the
// backend dev server runs on a non-default port.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');
const PROJECT_ROOT = fileURLToPath(new URL('.', import.meta.url));

const ROUTE_SHARED_MODULES = [
  '/features/models/data/modelLoadStore.ts',
  '/features/models/index.ts',
  '/features/models/ui/ModelsPage.tsx',
  '/features/nodes/data/nodeExecutionStore.ts',
  '/features/nodes/index.ts',
  '/features/nodes/ui/NodesPage.tsx',
  '/platform/browser/downloadBlob.ts',
  '/platform/query/client.ts',
  '/platform/time/serverTimestamp.ts',
  '/platform/transport/connectionStore.ts',
  '/platform/transport/socketHub.ts',
  '/platform/ui/theme/applyTheme.ts',
  '/workbench/components/WorkbenchSplashScreen.tsx',
  '/workbench/hotkeys/catalog.ts',
  '/workbench/launchpad/formatRelativeTime.ts',
  // The Launchpad writes `?intent=` and the editor's session controller reads
  // it. Without this the editor pulls the whole Launchpad chunk for a lookup
  // table — a 66 KB, one-extra-request regression on the editor route.
  '/workbench/launchpad/intents.ts',
  '/workbench/palette/settingsEntryDeps.ts',
  '/workbench/projects/ids.ts',
  '/workbench/projects/library.ts',
  '/workbench/projects/projectFile.ts',
  '/workbench/settings/SettingsDialogHost.tsx',
] as const;

const WORKBENCH_TOPBAR_MODULES = [
  '/workbench/shell/topbar/LayoutPresetAdminDialogs.tsx',
  '/workbench/shell/topbar/LayoutPresetStrip.tsx',
  '/workbench/shell/topbar/ProjectSwitcher.tsx',
] as const;

const matchesAnySuffix = (id: string, suffixes: readonly string[]) => suffixes.some((suffix) => id.endsWith(suffix));

const getLegacyChunkName = (id: string): string | null => {
  if (
    matchesAnySuffix(id, [
      '/platform/state/selectors.ts',
      '/workbench/palette/paletteStore.ts',
      '/platform/search/dateTokens.ts',
      '/platform/performance/semanticReady.ts',
    ])
  ) {
    return 'shared';
  }

  if (
    matchesAnySuffix(id, [
      '/platform/i18n/client.ts',
      '/platform/react/useMountEffect.ts',
      '/platform/ui/theme/system.ts',
      '/workbench/hotkeys/resolve.ts',
      '/workbench/settings/settingsDialogStore.ts',
    ])
  ) {
    return 'shell-shared';
  }

  if (matchesAnySuffix(id, ['/features/gallery/core/items.ts', '/features/gallery/ui/galleryStateView.ts'])) {
    return 'gallery-state';
  }

  if (!id.includes('/node_modules/')) {
    return null;
  }

  if (
    id.includes('/node_modules/ag-psd/') ||
    id.includes('/node_modules/pako/') ||
    id.includes('/node_modules/base64-js/')
  ) {
    return 'ag-psd';
  }

  if (id.includes('/node_modules/yaml/')) {
    return 'yaml';
  }

  if (id.includes('/node_modules/react-icons/')) {
    return 'react-icons';
  }

  if (id.includes('/node_modules/@xyflow/') || /\/node_modules\/d3-[^/]+\//.test(id)) {
    return 'workflow-vendor';
  }

  if (id.includes('/node_modules/perfect-freehand/') || id.includes('/node_modules/@dnd-kit/')) {
    return 'editor-interactions';
  }

  if (id.includes('/node_modules/@chakra-ui/') || id.includes('/node_modules/@emotion/')) {
    return 'chakra';
  }

  return 'vendor';
};

export default defineConfig({
  base: './',
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              includeDependenciesRecursively: false,
              name: 'route-shared',
              priority: 30,
              test: (id) => matchesAnySuffix(id, ROUTE_SHARED_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'workbench-topbar',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WORKBENCH_TOPBAR_MODULES),
            },
            {
              name: getLegacyChunkName,
            },
          ],
        },
      },
      preserveEntrySignatures: 'allow-extension',
    },
  },
  plugins: [
    react(),
    babel({
      presets: [reactCompilerPreset()],
    }),
    chunkSourceManifest({ projectRoot: PROJECT_ROOT }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@app': fileURLToPath(new URL('./src/app', import.meta.url)),
      '@assets': fileURLToPath(new URL('./src/assets', import.meta.url)),
      '@features': fileURLToPath(new URL('./src/features', import.meta.url)),
      '@platform': fileURLToPath(new URL('./src/platform', import.meta.url)),
      '@theme': fileURLToPath(new URL('./src/platform/ui/theme', import.meta.url)),
      '@workbench': fileURLToPath(new URL('./src/workbench', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    proxy: {
      '/api/': {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        target: `${BACKEND_URL}/api/`,
      },
      '/openapi.json': {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/openapi.json/, ''),
        target: `${BACKEND_URL}/openapi.json`,
      },
      '/ws/socket.io': {
        target: BACKEND_WS_URL,
        ws: true,
      },
    },
  },
});
