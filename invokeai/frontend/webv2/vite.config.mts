import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

import { chunkSourceManifest } from './scripts/chunk-source-manifest.mjs';
import { localeAssetsPlugin } from './scripts/locale-assets-plugin.mjs';
import { serviceWorkerPlugin } from './scripts/service-worker-plugin.mjs';

// INVOKEAI_DEV_BACKEND=http://127.0.0.1:9091 points at a non-default backend.
const BACKEND_URL = process.env.INVOKEAI_DEV_BACKEND ?? 'http://127.0.0.1:9090';
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, 'ws');

// INVOKEAI_DEV_HOSTS=my-box.local,10.0.0.5 allows other hostnames.
const ALLOWED_HOSTS = process.env.INVOKEAI_DEV_HOSTS?.split(',')
  .map((host) => host.trim())
  .filter(Boolean);
const PROJECT_ROOT = fileURLToPath(new URL('.', import.meta.url));

// Group eager dependencies shared by both routes to avoid extra chunk requests.
const ROUTE_SHARED_MODULES = [
  '/features/fonts/data/keys.ts',
  '/features/fonts/launchpad.tsx',
  '/features/fonts/react.tsx',
  '/features/fonts/runtime.ts',
  '/features/models/data/modelLoadStore.ts',
  '/features/models/index.ts',
  '/features/models/ui/ModelsPage.tsx',
  '/features/nodes/data/nodeExecutionStore.ts',
  '/features/nodes/index.ts',
  '/features/nodes/ui/NodesPage.tsx',
  '/platform/browser/downloadBlob.ts',
  '/platform/ui/BrandIcon.tsx',
  '/platform/ui/Button.tsx',
  '/platform/ui/Tooltip.tsx',
  '/platform/ui/RetryBoundary.tsx',
  '/platform/ui/PanelHeader.tsx',
  '/platform/ui/settings/contracts.ts',
  '/platform/core/concurrency.ts',
  '/platform/query/client.ts',
  '/platform/time/serverTimestamp.ts',
  '/platform/transport/connectionStore.ts',
  '/platform/transport/socketHub.ts',
  '/platform/ui/ConfirmDialog.tsx',
  '/platform/ui/MiddleTruncate.tsx',
  '/platform/ui/theme/applyTheme.ts',
  '/workbench/components/WorkbenchSplashScreen.tsx',
  '/workbench/hotkeys/catalog.ts',
  '/workbench/hotkeys/modalLayer.ts',
  '/workbench/launchpad/formatRelativeTime.ts',
  // Without this the editor pulls the whole Launchpad chunk for one lookup table.
  '/workbench/launchpad/intents.ts',
  '/workbench/mediaReferences.ts',
  '/workbench/palette/settingsEntryDeps.ts',
  '/workbench/projects/covers.ts',
  '/workbench/projects/components/ProjectFileOptionsProvider.tsx',
  '/workbench/projects/ids.ts',
  '/workbench/projects/invk/format.ts',
  '/workbench/projects/library.ts',
  '/workbench/projects/projectAssets.ts',
  '/workbench/projects/projectFile.ts',
  '/workbench/projects/projectFileErrors.ts',
  '/workbench/projects/projectFileToasts.ts',
  '/workbench/projects/useProjectFileActions.ts',
  '/workbench/settings/SettingsDialogHost.tsx',
] as const;

// Group dependencies shared by the editor shell and lazy widgets to reduce boot requests.
const EDITOR_BOOT_SHARED_MODULES = [
  '/app/GalleryUiAdapter.tsx',
  '/features/gallery/picker.ts',
  '/features/gallery/ui/GalleryBoardCover.tsx',
  '/features/gallery/ui/GalleryBoardRowShell.tsx',
  '/features/gallery/ui/GalleryItemSearch.tsx',
  '/features/gallery/ui/GallerySearchField.tsx',
  '/features/gallery/ui/GalleryTileFrame.tsx',
  '/features/gallery/ui/GalleryUploadButton.tsx',
  '/features/gallery/ui/GalleryViewTabs.tsx',
  '/features/gallery/ui/GalleryWidgetContext.tsx',
  '/features/gallery/ui/galleryBoardGroups.ts',
  '/features/gallery/ui/galleryBoardLabels.ts',
  '/features/gallery/ui/galleryGridLayout.ts',
  '/features/gallery/ui/picker/GalleryPickerPopover.tsx',
  '/features/gallery/ui/useGalleryData.ts',
  '/features/gallery/ui/useGalleryUploadAction.ts',
  '/features/gallery/ui/useGalleryUploadInput.ts',
  '/features/generation/core/prompt/ast.ts',
  '/features/generation/core/prompt/attention.ts',
  '/features/generation/data/architectureCapabilitiesApi.ts',
  '/features/generation/data/architectureCapabilitiesStore.ts',
  '/features/generation/data/dynamicPromptsQueries.ts',
  '/features/generation/data/promptTemplates.ts',
  '/features/generation/data/promptUtilities.ts',
  '/features/generation/data/systemPrompts.ts',
  '/features/generation/prompts.ts',
  '/features/generation/queries.ts',
  '/features/generation/runtime.ts',
  '/features/generation/ui/promptFields/promptAttentionHotkeys.ts',
  '/platform/ui/SeedInput.tsx',
  '/workbench/shell/topbar/LayoutPresetAdminDialogs.tsx',
  '/workbench/shell/topbar/LayoutPresetStrip.tsx',
  '/workbench/shell/topbar/ProjectSwitcher.tsx',
] as const;

// Keep widget metadata separate so Launchpad settings cannot import editor boot UI.
const WIDGET_METADATA_MODULES = [
  '/features/gallery/settingsContribution.ts',
  '/features/queue/widget.ts',
  '/features/workflow/widget.ts',
  '/workbench/settings/applicationContributions.ts',
  '/workbench/settings/catalog.ts',
  '/workbench/widgets/canvas/canvasSettings.ts',
  '/workbench/widgets/canvas/settingsContribution.ts',
  '/workbench/widgets/image-map/settingsContribution.ts',
  '/workbench/widgets/layers/panes/editorPaneLayout.ts',
  '/workbench/widgets/manifests.ts',
  '/workbench/widgets/preview/settingsContribution.ts',
  '/workbench/widgets/preview/previewSettings.ts',
] as const;

// Gallery's shared state projection and UI port travel together.
const GALLERY_STATE_MODULES = [
  '/features/gallery/core/items.ts',
  '/features/gallery/core/recentImages.ts',
  '/features/gallery/core/selection.ts',
  '/features/gallery/core/semanticImageQuery.ts',
  '/features/gallery/core/settings.ts',
  '/features/gallery/ui/GalleryUiContext.tsx',
  '/features/gallery/ui/galleryStateView.ts',
  '/features/queue/contracts.ts',
  '/features/queue/core/generationMeta.ts',
  '/features/queue/core/historySnapshot.ts',
  '/features/queue/core/historySummary.ts',
  '/features/queue/core/progressRail.ts',
  '/features/queue/core/submissionRules.ts',
  '/features/queue/data/events.ts',
] as const;

// Group boot-mounted widget hosts and their shared helpers; Launchpad must not load this chunk.
const WIDGET_HOST_MODULES = [
  '/platform/react/focusIfUnclaimed.ts',
  '/features/queue/ui/QueueDataRuntime.tsx',
  '/features/workflow/ui/WorkflowWidgetChrome.tsx',
  '/workbench/widgets/image-map/ImageMapDataRuntime.tsx',
] as const;

// Keep the Image Map data modules in one chunk. Its API is also reached through the Gallery's lazily loaded label
// cache, and without this group Rolldown splits the API out, costing every editor boot a request.
const IMAGE_MAP_DATA_MODULES = [
  '/workbench/image-map/api.ts',
  '/workbench/image-map/imageMapStore.ts',
  '/workbench/image-map/indexProgress.ts',
] as const;

// Group Canvas/Layer interaction dependencies to avoid extra text-tool activation requests.
const CANVAS_LAYER_SHARED_MODULES = [
  '/features/workflow/core/layerWorkflow.ts',
  '/workbench/canvas-operations/react.ts',
  '/workbench/canvas-operations/useCanvasEngine.ts',
  '/workbench/canvasProjectMutationPort.ts',
  '/workbench/useCanvasProjectMutationDispatch.ts',
  '/workbench/widgets/canvas/canvasInteractionLock.ts',
  '/workbench/widgets/canvas/color-system/colorPair.ts',
  '/workbench/widgets/canvas/color-system/useActiveColors.ts',
  '/workbench/widgets/canvas/engineStoreHooks.ts',
  '/workbench/widgets/canvas/textFontStyle.tsx',
  '/workbench/widgets/canvas/tool-presentation/FormControls.tsx',
  '/workbench/widgets/canvas/tool-presentation/PropertyPrimitives.tsx',
  '/workbench/widgets/canvas/tool-presentation/propertyGroupStore.ts',
  '/workbench/widgets/canvas/useCanvasEngine.ts',
  '/workbench/widgets/canvas/useColorSampler.ts',
  '/workbench/widgets/canvas/useStructuralCommit.ts',
  '/workbench/widgets/layers/LayerContextMenu.tsx',
  '/workbench/widgets/layers/RunLayerWorkflowDialog.tsx',
  '/workbench/widgets/layers/colorLabels.ts',
  '/workbench/widgets/layers/layerActionSession.ts',
  '/workbench/widgets/layers/layerContextActions.ts',
  '/workbench/widgets/layers/layerContextMenuLayout.ts',
  '/workbench/widgets/layers/layerExportActions.ts',
  '/workbench/widgets/layers/layerGroupCommands.ts',
  '/workbench/widgets/layers/layerMenuState.ts',
  '/workbench/widgets/layers/layerPropertiesRequestStore.ts',
  '/workbench/widgets/layers/runLayerWorkflow.ts',
  '/workbench/widgets/layers/useSelectedModelBase.ts',
];

const matchesAnySuffix = (id: string, suffixes: readonly string[]) => suffixes.some((suffix) => id.endsWith(suffix));

const getLegacyChunkName = (id: string): string | null => {
  if (
    matchesAnySuffix(id, [
      '/platform/state/selectors.ts',
      '/workbench/palette/paletteStore.ts',
      '/platform/search/dateTokens.ts',
      '/platform/performance/semanticReady.ts',
      // A pure leaf every seeded owner imports; on its own it would cost the editor boot a request.
      '/platform/core/seed.ts',
    ])
  ) {
    return 'shared';
  }

  if (
    matchesAnySuffix(id, [
      '/platform/i18n/client.ts',
      '/platform/i18n/languages.ts',
      '/platform/react/useMountEffect.ts',
      '/platform/ui/theme/system.ts',
      '/workbench/hotkeys/resolve.ts',
      '/workbench/settings/settingsDialogStore.ts',
    ])
  ) {
    return 'shell-shared';
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
  define: {
    // Use a boolean: Vitest browser mode treats string defines as truthy string literals.
    __CANVAS_GOLDEN_UPDATE__: false,
  },
  base: './',
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              includeDependenciesRecursively: false,
              name: 'canvas-layer-shared',
              priority: 30,
              test: (id) => matchesAnySuffix(id, CANVAS_LAYER_SHARED_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'gallery-state',
              priority: 30,
              test: (id) => matchesAnySuffix(id, GALLERY_STATE_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'route-shared',
              priority: 30,
              test: (id) => matchesAnySuffix(id, ROUTE_SHARED_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'editor-boot-shared',
              priority: 30,
              test: (id) => matchesAnySuffix(id, EDITOR_BOOT_SHARED_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'widget-metadata',
              priority: 30,
              test: (id) =>
                matchesAnySuffix(id, WIDGET_METADATA_MODULES) || /\/workbench\/widgets\/[^/]+\/manifest\.ts$/.test(id),
            },
            {
              includeDependenciesRecursively: false,
              name: 'widget-hosts',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WIDGET_HOST_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'imageMapStore',
              priority: 30,
              test: (id) => matchesAnySuffix(id, IMAGE_MAP_DATA_MODULES),
            },
            {
              // ~1 MB, only the lazy Image Map plot needs it.
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
    localeAssetsPlugin({ projectRoot: PROJECT_ROOT }),
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
