import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

import { chunkSourceManifest } from './scripts/chunk-source-manifest.mjs';
import { serviceWorkerPlugin } from './scripts/service-worker-plugin.mjs';

// Override with e.g. INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 when the
// backend dev server runs on a non-default port.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');

// Set e.g. INVOKEAI_DEV_HOSTS=my-box.local,10.0.0.5 when reaching the dev
// server by a hostname other than localhost.
const ALLOWED_HOSTS = process.env.INVOKEAI_DEV_HOSTS?.split(',')
  .map((host) => host.trim())
  .filter(Boolean);
const PROJECT_ROOT = fileURLToPath(new URL('.', import.meta.url));

const ROUTE_SHARED_MODULES = [
  '/features/models/data/modelLoadStore.ts',
  '/features/models/index.ts',
  '/features/models/ui/ModelsPage.tsx',
  '/features/nodes/data/nodeExecutionStore.ts',
  '/features/nodes/index.ts',
  '/features/nodes/ui/NodesPage.tsx',
  '/platform/browser/downloadBlob.ts',
  '/platform/core/concurrency.ts',
  '/platform/query/client.ts',
  '/platform/time/serverTimestamp.ts',
  '/platform/transport/connectionStore.ts',
  '/platform/transport/socketHub.ts',
  // Both routes confirm destructive actions (deleting a project, discarding
  // an import) with the same dialog. Splitting the widget hosts out of their
  // view chunks gave it a second, editor-only consumer alongside the
  // Launchpad's — without grouping it crosses the automatic chunking
  // algorithm's single-consumer threshold and gets extracted into its own
  // file on both routes, in place of the free inline copies each had before.
  '/platform/ui/ConfirmDialog.tsx',
  // Name/identifier truncation used eagerly by Launchpad project cards and
  // editor widget chrome alike.
  '/platform/ui/MiddleTruncate.tsx',
  '/platform/ui/theme/applyTheme.ts',
  '/workbench/components/WorkbenchSplashScreen.tsx',
  '/workbench/hotkeys/catalog.ts',
  '/workbench/launchpad/formatRelativeTime.ts',
  // The Launchpad writes `?intent=` and the editor's session controller reads
  // it. Without this the editor pulls the whole Launchpad chunk for a lookup
  // table — a 66 KB, one-extra-request regression on the editor route.
  '/workbench/launchpad/intents.ts',
  '/workbench/palette/settingsEntryDeps.ts',
  '/workbench/projects/covers.ts',
  '/workbench/projects/ids.ts',
  // The `.invk` surface both routes touch eagerly: the extension for the file
  // picker, and the error class every import call site catches to translate.
  // The schema, the ZIP codec and the archive itself stay behind lazy imports.
  '/workbench/projects/invk/format.ts',
  '/workbench/projects/library.ts',
  // Editor-eager through `syncedPersistence` (cover selection on every save)
  // and Launchpad-eager through the import workflow. Pulling it out of the
  // shared chunk to spare the Launchpad ~1.5 KB cost the editor a whole extra
  // request, because nothing else would then group it — a round trip is the
  // worse end of that trade.
  '/workbench/projects/projectAssets.ts',
  '/workbench/projects/projectFile.ts',
  '/workbench/projects/projectFileErrors.ts',
  // Both routes offer Import and Export, so both need the reporter and the
  // hooks that drive it.
  '/workbench/projects/projectFileToasts.ts',
  '/workbench/projects/useProjectFileActions.ts',
  '/workbench/settings/SettingsDialogHost.tsx',
] as const;

const WORKBENCH_TOPBAR_MODULES = [
  '/workbench/shell/topbar/LayoutPresetAdminDialogs.tsx',
  '/workbench/shell/topbar/LayoutPresetStrip.tsx',
  '/workbench/shell/topbar/ProjectSwitcher.tsx',
] as const;

// Modules the always-static editor shell and the boot-time `widget-hosts`
// chunk both reach for. Editor-only — folding them into `route-shared`
// would cost the Launchpad code it never loads, the trade the
// `route-shared` comments above warn against. Splitting the widget hosts
// out of their view chunks (see `WIDGET_HOST_MODULES`) gave each of these a
// second, independent consumer; without grouping, that crosses the
// automatic chunking algorithm's single-consumer threshold and each gets
// extracted into its own file — several extra editor-only requests for code
// that was previously duplicated inline for free. Riding the existing
// `workbench-topbar` chunk rather than a new one costs it bytes, not a
// request: `workbench-topbar` is already fetched by every editor boot.
const EDITOR_SHARED_RUNTIME_MODULES = [
  '/features/queue/core/graphInputMedia.ts',
  '/features/queue/core/progressImage.ts',
  '/features/queue/data/activeProgressTargetStore.ts',
  '/features/queue/data/generationDevicesStore.ts',
  '/features/queue/data/itemProgressStore.ts',
  '/features/queue/data/progressImageStore.ts',
  '/features/queue/data/progressStore.ts',
  '/features/queue/data/realtimeRuntime.ts',
  '/features/queue/publicApi.ts',
  '/features/queue/runtime.ts',
  '/features/queue/runtime/coordinator.ts',
  '/features/queue/ui/queueConfirmationStore.ts',
  '/features/workflow/core/validation.ts',
] as const;

// The singleton widget hosts the editor mounts once at boot: workflow's
// dialog shell, queue's data runtime, image-map's data runtime. Each is
// fetched together with the other two on every editor boot, all three are
// always needed, and none is ever needed without the others — splitting
// them into three separate chunks (one per widget's `loadHost`) traded a
// shared-chunk request for a per-widget one three times over. Grouping them
// back into a single chunk keeps the per-host code-splitting boundary (so a
// host still never drags its widget's view chunk along) while paying for
// that boundary once instead of three times.
const WIDGET_HOST_MODULES = [
  '/features/queue/ui/QueueDataRuntime.tsx',
  '/features/workflow/ui/WorkflowWidgetChrome.tsx',
  '/workbench/widgets/image-map/ImageMapDataRuntime.tsx',
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

  // Only `projects/invk/archive.ts` reaches for this, and only when a project
  // file is actually read or written — the same treatment ag-psd gets.
  if (id.includes('/node_modules/fflate/')) {
    return 'fflate';
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
              test: (id) =>
                matchesAnySuffix(id, WORKBENCH_TOPBAR_MODULES) || matchesAnySuffix(id, EDITOR_SHARED_RUNTIME_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'widget-hosts',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WIDGET_HOST_MODULES),
            },
            {
              // Plotly is large (~1MB min) and only used by the lazy-loaded
              // Image Map plot; keep it out of the eager vendor chunk.
              name: 'plotly',
              priority: 30,
              test: (id) => id.includes('plotly') && id.includes('node_modules'),
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
    serviceWorkerPlugin({ projectRoot: PROJECT_ROOT }),
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
    allowedHosts: ALLOWED_HOSTS,
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
