import babel from '@rolldown/plugin-babel';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
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

// The Python package owns the release version; fail the build rather than ship a guessed one.
const VERSION_FILE = fileURLToPath(new URL('../../version/invokeai_version.py', import.meta.url));
const readAppVersion = (): string => {
  const version = /^__version__\s*=\s*["']([^"']+)["']/m.exec(readFileSync(VERSION_FILE, 'utf8'))?.[1];

  if (!version) {
    throw new Error(`No __version__ assignment found in ${VERSION_FILE}.`);
  }

  return version;
};

// Group eager dependencies shared by both routes to avoid extra chunk requests.
const ROUTE_SHARED_MODULES = [
  '/features/fonts/data/keys.ts',
  '/features/intermediates/data/focus.ts',
  '/features/fonts/launchpad.tsx',
  '/features/fonts/react.tsx',
  '/features/fonts/runtime.ts',
  '/features/models/data/modelLoadStore.ts',
  '/features/models/index.ts',
  '/features/models/ui/add-models/starterModelInstallSources.ts',
  '/features/models/ui/ModelsPage.tsx',
  '/features/nodes/data/nodeExecutionStore.ts',
  '/features/nodes/index.ts',
  '/features/nodes/ui/NodesPage.tsx',
  '/features/gallery/data/imageRecovery.ts',
  '/features/gallery/ui/useGalleryImageRecovery.ts',
  '/platform/browser/downloadBlob.ts',
  '/platform/browser/imeComposition.ts',
  '/platform/browser/searchShortcut.ts',
  '/platform/browser/webLocks.ts',
  '/workbench/projects/editorSession.ts',
  '/platform/ui/BrandIcon.tsx',
  '/platform/ui/Button.tsx',
  '/platform/ui/DropZone.tsx',
  '/features/identity/core/schemas.ts',
  '/features/identity/core/sessionPolicy.ts',
  '/features/identity/core/storedCredential.ts',
  '/features/identity/core/tokenStorage.ts',
  '/features/identity/data/api.ts',
  '/features/identity/session.ts',
  '/features/identity/ui/AuthScreen.tsx',
  '/features/identity/ui/PasswordInput.tsx',
  '/features/identity/ui/useIdentityNotify.ts',
  '/platform/react/useZodForm.ts',
  '/platform/ui/InvokeMark.tsx',
  '/platform/ui/EmptyState.tsx',
  '/platform/browser/clipboard.ts',
  '/platform/ui/JsonPreview.tsx',
  '/platform/ui/Tabs.tsx',
  '/platform/ui/toaster.tsx',
  '/platform/ui/Slider.tsx',
  '/platform/ui/SliderNumberField.tsx',
  '/platform/ui/Tooltip.tsx',
  '/platform/ui/Menu.tsx',
  '/platform/ui/Popover.tsx',
  '/platform/ui/widgetOverlayRegistry.ts',
  '/platform/ui/widgetOverlays.ts',
  '/platform/ui/RetryBoundary.tsx',
  '/platform/ui/PanelHeader.tsx',
  '/platform/ui/settings/contracts.ts',
  '/platform/core/concurrency.ts',
  '/platform/query/client.ts',
  '/platform/runtime/appMetadata.ts',
  '/workbench/shell/useWhatsNew.ts',
  '/workbench/shell/whatsNewStore.ts',
  '/platform/time/serverTimestamp.ts',
  '/platform/transport/connectionStore.ts',
  '/platform/transport/socketHub.ts',
  '/platform/ui/ConfirmDialog.tsx',
  '/platform/ui/Dialog.tsx',
  '/platform/ui/modalPresence.ts',
  '/platform/react/useExitRetainedValue.ts',
  '/platform/ui/MiddleTruncate.tsx',
  '/platform/ui/list/ListDivider.tsx',
  '/platform/ui/list/ListItem.tsx',
  '/platform/ui/list/ListSectionHeader.tsx',
  '/platform/ui/list/ListStack.tsx',
  '/platform/ui/list/listLayout.ts',
  '/platform/ui/list/virtualSections.ts',
  '/platform/react/usePreservedScrollOffset.ts',
  '/platform/ui/Scrollable.tsx',
  '/platform/ui/useScrollAreaPhantomHeal.ts',
  '/platform/ui/theme/applyTheme.ts',
  '/workbench/components/WorkbenchSplashScreen.tsx',
  '/assets/SplashImage.webp',
  '/workbench/hotkeys/catalog.ts',
  '/workbench/hotkeys/modalLayer.ts',
  '/workbench/launchpad/formatRelativeTime.ts',
  '/workbench/launchpad/StayOnLaunchpadButton.tsx',
  '/workbench/launchpad/startupLaunch.ts',
  '/workbench/projects/openProjects.ts',
  '/workbench/projects/projectFlush.ts',
  '/workbench/projects/syncStore.ts',
  '/workbench/launchpad/intents.ts',
  '/workbench/mediaReferences.ts',
  '/workbench/palette/settingsEntryDeps.ts',
  '/workbench/projects/covers.ts',
  '/workbench/projects/components/ProjectFileOptionsProvider.tsx',
  '/workbench/projects/ids.ts',
  '/workbench/projects/invk/format.ts',
  '/workbench/projects/library.ts',
  '/workbench/projects/projectAssets.ts',
  '/workbench/projects/projectHalves.ts',
  '/workbench/projects/projectFile.ts',
  '/workbench/projects/projectFileErrors.ts',
  '/workbench/projects/projectFileToasts.ts',
  '/workbench/projects/session.ts',
  '/workbench/projects/useProjectFileActions.ts',
  '/workbench/settings/SettingsDialogHost.tsx',
  '/workbench/settings/launchpad.tsx',
  '/workbench/settings/projectSettings.ts',
  '/workbench/settings/settingsSearchShortcut.ts',
] as const;

// Group dependencies shared by the editor shell and lazy widgets to reduce boot requests.
const EDITOR_BOOT_SHARED_MODULES = [
  '/workbench/focusRegions.tsx',
  '/workbench/hotkeys/hintSources.tsx',
  '/workbench/hotkeys/shortcutHints.ts',
  '/workbench/hotkeys/targetWidget.ts',
  '/workbench/hotkeys/useShortcutBinding.ts',
  '/workbench/widget-frame/compactWidgetCapacity.tsx',
  '/platform/react/isEventFromOwnDom.ts',
  '/features/gallery/ui/GalleryItemSearch.tsx',
  '/workbench/image-actions/GalleryBoardDestinationGroups.tsx',
  '/workbench/widgets/layers/layerTypeIcon.ts',
  '/platform/ui/ScrollableMenuContent.tsx',
  '/app/GalleryUiAdapter.tsx',
  '/features/generation/core/prompt/ast.ts',
  '/features/generation/core/prompt/attention.ts',
  '/features/generation/data/dynamicPromptsQueries.ts',
  '/features/generation/data/promptTemplates.ts',
  '/features/generation/data/promptUtilities.ts',
  '/features/generation/data/systemPrompts.ts',
  '/features/generation/prompts.ts',
  '/features/generation/queries.ts',
  '/features/generation/runtime.ts',
  '/features/generation/ui/promptFields/promptAttentionHotkeys.ts',
  '/platform/ui/ResizeHandle.tsx',
  '/platform/ui/SeedInput.tsx',
  '/platform/ui/ScrubberField.tsx',
  '/workbench/shell/topbar/LayoutPresetAdminDialogs.tsx',
  '/workbench/shell/topbar/LayoutPresetStrip.tsx',
  '/workbench/shell/topbar/ProjectSwitcher.tsx',
  '/workbench/widgets/canvas/invoke/canvasScaling.ts',
  '/workbench/widgets/canvas/invoke/canvasStrength.ts',
  '/workbench/widgets/canvas/bboxGrid.ts',
  '/workbench/settings/aspectRatioPresets.ts',
  '/workbench/widgets/shared/AutoSwitchMenu.tsx',
  '/platform/ui/ProgressBadge.tsx',
  '/workbench/widgets/layers/layerPropertiesRequestStore.ts',
  '/workbench/widgets/layers/revealLayerProperties.ts',
  '/workbench/useOpenWorkbenchWidget.ts',
  '/features/queue/reveal.ts',
  '/features/queue/ui/queueUiStore.ts',
  '/workbench/shell/notifications/editorNotificationActions.ts',
  '/workbench/shell/notifications/notificationActions.ts',
  '/workbench/shell/notifications/notificationToasts.ts',
  '/workbench/shell/notifications/toastPolicy.ts',
  '/workbench/image-actions/useFindGalleryItem.ts',
  '/workbench/queue-integration/queueRecallCache.ts',
  '/platform/react/useModifierHeld.ts',
  '/platform/browser/dropZones.ts',
  '/platform/react/useNativeFileDrag.ts',
  '/workbench/widgets/shared/DropZoneLabel.tsx',
  '/workbench/widgets/shared/useDroppedFileUpload.ts',
] as const;

// Editor runtime the workbench store reaches; ungrouped, Rolldown splits it by importer set into extra boot requests.
const WORKBENCH_RUNTIME_SHARED_MODULES = [
  // Layer factories and engine actions share the adapter helper; keep it in the existing boot request.
  '/workbench/controlAdapters.ts',
  // The document model and engine actions share these boot primitives.
  '/workbench/canvas-engine/contracts.ts',
  '/workbench/canvas-engine/document-model/documentModel.ts',
  '/workbench/canvas-engine/document-model/postconditions.ts',
  '/workbench/canvas-engine/document-model/semanticLeaf.ts',
  '/workbench/canvas-engine/document-model/semanticNode.ts',
  '/workbench/canvas-engine/document/documentIndex.ts',
  '/workbench/canvas-engine/document/documentTree.ts',
  '/workbench/canvas-engine/document/insertionAnchors.ts',
  '/workbench/canvas-engine/document/layerEligibility.ts',
  '/workbench/canvas-engine/document/layerStacks.ts',
  '/workbench/canvas-engine/document/selectionRepair.ts',
  '/workbench/canvas-engine/document/sources.ts',
  '/workbench/canvas-engine/layerExportGuards.ts',
  '/workbench/canvas-engine/math/mat2d.ts',
  '/workbench/canvas-engine/math/rect.ts',
  '/workbench/canvas-engine/math/snapping.ts',
  '/workbench/canvas-engine/mutationContracts.ts',
  '/workbench/canvas-engine/render/rasterizers/textRasterizer.ts',
  '/workbench/canvas-engine/transform/transformMath.ts',
  '/platform/react/draftRegistry.ts',
  '/features/generation/core/projectPromptDraft.ts',
  '/features/generation/core/prompt/compel.ts',
  // The editor boot and the lazy Model Manager (both routes, which already loads this chunk) read the architecture
  // table's store. `editor-boot-shared` would drag the editor shell into the manager; split out, the store and its
  // entry cost the editor boot requests.
  '/features/models/architecturesStore.ts',
  '/features/models/data/architectureCapabilitiesApi.ts',
  '/features/models/data/architectureCapabilitiesStore.ts',
  '/features/queue/core/deviceLabels.ts',
  '/features/queue/devices.ts',
  '/features/queue/core/graphInputMedia.ts',
  '/features/queue/data/activeProgressTargetStore.ts',
  '/features/queue/data/progressImageStore.ts',
  '/features/queue/data/progressStore.ts',
  '/features/queue/data/realtimeRuntime.ts',
  '/features/queue/publicApi.ts',
  '/features/queue/runtime.ts',
  '/features/queue/runtime/cancelRequests.ts',
  '/features/queue/runtime/coordinator.ts',
  '/features/queue/runtime/deliveryClaims.ts',
  '/features/queue/runtime/receiptAcknowledgements.ts',
  '/features/queue/ui/useDeviceLabel.ts',
  '/features/upscale/core/settings.ts',
  '/features/upscale/ui/UpscaleUiContext.tsx',
  '/features/video/core/dimensions.ts',
  '/features/video/core/settings.ts',
  '/features/video/core/videoPolicies.ts',
  '/features/video/core/widgetValues.ts',
  '/features/video/ui/VideoUiContext.tsx',
  '/features/workflow/core/buildGraph.ts',
  '/features/workflow/core/formFieldSettings.ts',
  '/platform/react/useDebouncedValue.ts',
] as const;

// The editor boots all of these and the palette reaches the barrel lazily, so splitting adds requests only.
const WORKFLOW_CORE_MODULES = [
  '/features/workflow/core/batch.ts',
  '/features/workflow/core/callSavedWorkflow.ts',
  '/features/workflow/core/connectors.ts',
  '/features/workflow/core/document.ts',
  '/features/workflow/core/fields.ts',
  '/features/workflow/core/forLoops.ts',
  '/features/workflow/core/graphIndex.ts',
  '/features/workflow/core/readinessIssues.ts',
  '/features/workflow/core/types.ts',
  '/features/workflow/core/validation.ts',
  '/features/workflow/core/workflowJson.ts',
  '/features/workflow/data/templates.ts',
  '/features/workflow/react.ts',
  '/features/workflow/ui/WorkflowUiContext.tsx',
  '/features/workflow/ui/editor/selectionStore.ts',
  '/features/workflow/ui/workflowUiStore.ts',
  '/features/workflow/utility.ts',
  '/workbench/projectWorkflows.ts',
  // Workflow's model requirements and Generation's policies read Models' pure architecture entry, and Related Models
  // (in the lazy Model Manager) the table registry behind it. The editor boot and the lazy Model and Node Managers all
  // load this chunk already; split out, the two modules cost the editor boot and each manager requests.
  '/features/models/architectures.ts',
  '/features/models/core/architectureCapabilities.ts',
] as const;

// The editor's boot hosts and the lazy library dialog share the transport; split out, it costs every boot a request.
const WORKFLOW_QUERIES_MODULES = [
  '/features/workflow/data/api.ts',
  '/features/workflow/data/libraryCache.ts',
  '/features/workflow/data/publication.ts',
  '/features/workflow/data/savedWorkflowQueries.ts',
  '/features/workflow/queries.ts',
] as const;

// Keep widget metadata separate so Launchpad settings cannot import editor boot UI.
const WIDGET_METADATA_MODULES = [
  '/features/gallery/settingsContribution.ts',
  '/features/intermediates/settingsContribution.ts',
  '/features/queue/widget.ts',
  '/features/workflow/widget.ts',
  '/workbench/settings/applicationContributions.ts',
  '/workbench/settings/catalog.ts',
  '/workbench/widgets/canvas/canvasSettings.ts',
  // canvasSettings reads these keys; grouped elsewhere they can close an import cycle with the editor context.
  '/workbench/widgets/canvas/invoke/canvasCompositing.ts',
  '/workbench/widgets/canvas/settingsContribution.ts',
  '/workbench/widgets/image-map/settingsContribution.ts',
  '/workbench/widgets/layers/panes/editorPaneLayout.ts',
  '/workbench/widgets/manifests.ts',
  '/workbench/widgets/preview/settingsContribution.ts',
  '/workbench/widgets/preview/previewSettings.ts',
] as const;

// The Launchpad's model manager uses the picker, so it fits neither editor-boot-shared nor gallery-state.
const GALLERY_PICKER_MODULES = [
  '/features/gallery/ui/galleryDnd.ts',
  '/features/gallery/utility.ts',
  '/features/gallery/ui/GalleryDragCursor.tsx',
  '/features/gallery/core/boardLabels.ts',
  '/features/gallery/core/mediaCounts.ts',
  '/features/gallery/data/queries.ts',
  '/features/gallery/data/queryCache.ts',
  '/features/gallery/data/uploadDestination.ts',
  '/features/gallery/ui/GallerySearchHelp.tsx',
  '/platform/state/compareAndSwapRollback.ts',
  '/features/gallery/picker.ts',
  '/features/gallery/ui/GalleryBoardCover.tsx',
  '/features/gallery/ui/GalleryBoardRowShell.tsx',
  '/features/gallery/ui/GalleryLoadError.tsx',
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
] as const;

// The backend and its URL helpers load together for recent outputs. Keep picker UI out of that lazy Launchpad read.
const GALLERY_BACKEND_MODULES = [
  '/features/gallery/core/imagePaths.ts',
  '/features/gallery/core/videoPaths.ts',
  '/features/gallery/data/backend.ts',
  '/features/gallery/data/imageUrls.ts',
  '/features/gallery/data/videoUrls.ts',
] as const;

// Gallery's shared state projection and UI port travel together.
const GALLERY_STATE_MODULES = [
  '/features/gallery/core/boardAccess.ts',
  '/features/gallery/core/items.ts',
  '/features/gallery/core/recentImages.ts',
  '/features/gallery/core/selection.ts',
  '/features/gallery/core/semanticImageQuery.ts',
  '/features/gallery/core/settings.ts',
  '/features/gallery/react.ts',
  '/features/gallery/ui/GalleryUiContext.tsx',
  '/platform/react/contextMenuPoint.ts',
  '/features/gallery/ui/GalleryMediaUsagePanel.tsx',
  '/features/gallery/ui/galleryStateView.ts',
  '/features/queue/contracts.ts',
  '/features/queue/core/itemLabels.ts',
  '/features/queue/core/queueActivity.ts',
  '/features/queue/core/retryReport.ts',
  '/features/queue/ui/queueOwnership.ts',
  '/features/queue/ui/queueActionRequests.ts',
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

// Boot image menus the lazy queue recall chunks also import; ungrouped, Rolldown splits them into a boot request.
const IMAGE_ACTIONS_MODULES = [
  '/workbench/canvas-operations/canvasImportNotice.ts',
  '/workbench/image-actions/ContextSubMenu.tsx',
  '/workbench/image-actions/ImageContextMenu.tsx',
  '/workbench/image-actions/ImageItemMenuGroups.tsx',
  '/workbench/image-actions/RecallActionButtons.tsx',
  '/workbench/image-actions/appendReferenceImage.ts',
  '/workbench/image-actions/canvasImportError.ts',
  '/workbench/image-actions/executeVideoRecall.ts',
  '/workbench/image-actions/executeImageRecall.ts',
  '/workbench/image-actions/imageRecall.ts',
  '/workbench/image-actions/recordedModel.ts',
  '/workbench/image-actions/galleryOptimisticRollback.ts',
  '/workbench/image-actions/index.ts',
  '/workbench/image-actions/pbrMapsGraph.ts',
  '/workbench/image-actions/useDeletionConfirmation.tsx',
  '/workbench/image-actions/useImageActions.ts',
  '/workbench/image-actions/useImageRecallCapabilities.ts',
  '/workbench/image-actions/videoRecall.ts',
] as const;

// The Gallery's lazy label cache also reaches the API; ungrouped, Rolldown splits it into a boot request.
const IMAGE_MAP_DATA_MODULES = [
  '/workbench/image-map/api.ts',
  '/workbench/image-map/imageMapStore.ts',
  '/workbench/image-map/indexProgress.ts',
] as const;

// Group Canvas/Layer interaction dependencies to avoid extra text-tool activation requests.
const CANVAS_LAYER_SHARED_MODULES = [
  '/workbench/canvas-operations/ControlDetectionCoordinator.ts',
  '/workbench/canvas-operations/FilterOperationCoordinator.ts',
  '/workbench/canvas-operations/SelectObjectCoordinator.ts',
  '/workbench/canvas-operations/applicationPort.ts',
  '/workbench/canvas-operations/controlDetectionRunner.ts',
  '/workbench/canvas-operations/controlLayerIntents.ts',
  '/workbench/canvas-operations/createCanvasEngine.ts',
  '/workbench/canvas-operations/filterOperationSession.ts',
  '/workbench/canvas-operations/layerFilterRunner.ts',
  '/workbench/canvas-operations/operationController.ts',
  '/workbench/canvas-operations/operationStores.ts',
  '/workbench/canvas-operations/toolMemory.ts',
  '/features/workflow/core/layerWorkflow.ts',
  '/workbench/canvas-operations/react.ts',
  '/workbench/canvas-operations/useCanvasEngine.ts',
  '/workbench/canvasProjectMutationPort.ts',
  '/workbench/useCanvasProjectMutationDispatch.ts',
  '/workbench/widgets/canvas/canvasRestorePoints.ts',
  '/workbench/widgets/canvas/useCanvasRecovery.ts',
  '/workbench/widgets/canvas/canvasToolCommands.ts',
  '/workbench/widgets/canvas/color-system/colorPair.ts',
  '/workbench/widgets/canvas/color-system/maskTintTarget.ts',
  '/workbench/widgets/canvas/color-system/useActiveColors.ts',
  '/workbench/widgets/canvas/color-system/useMaskTintEditor.ts',
  '/workbench/widgets/canvas/engineStoreHooks.ts',
  '/workbench/widgets/canvas/textFontStyle.tsx',
  '/workbench/widgets/canvas/tool-presentation/FormControls.tsx',
  '/workbench/widgets/canvas/tool-presentation/PropertyPrimitives.tsx',
  '/workbench/widgets/canvas/tool-presentation/propertyGroupStore.ts',
  '/workbench/widgets/canvas/useCanvasEngine.ts',
  '/workbench/widgets/canvas/useColorSampler.ts',
  '/workbench/widgets/canvas/useStructuralCommit.ts',
  '/workbench/widgets/layers/LayerContextMenu.tsx',
  '/workbench/widgets/layers/LayerSaveToBoardSubmenu.tsx',
  '/workbench/widgets/layers/RunLayerWorkflowDialog.tsx',
  '/workbench/widgets/layers/colorLabels.ts',
  '/workbench/widgets/layers/layerCommands.ts',
  '/workbench/widgets/layers/layerActionSession.ts',
  '/workbench/widgets/layers/layerChildRows.ts',
  '/workbench/widgets/layers/layerChildSelection.ts',
  '/workbench/widgets/layers/layerContextActions.ts',
  '/workbench/widgets/layers/layerContextMenuLayout.ts',
  '/workbench/widgets/layers/layerExportActions.ts',
  '/workbench/widgets/layers/layerMenuState.ts',
  '/workbench/widgets/layers/runLayerWorkflow.ts',
  '/workbench/widgets/layers/useSelectedModelBase.ts',
];

// These engine primitives also serve document hydration or image actions before Canvas opens.
const CANVAS_ENGINE_BOOT_MODULES = [
  '/workbench/canvas-engine/api.ts',
  '/workbench/canvas-engine/capabilities.ts',
  '/workbench/canvas-engine/document-model/screenComposition.ts',
  '/workbench/canvas-engine/document/controlDerivation.ts',
  '/workbench/canvas-engine/document/layerFactories.ts',
  '/workbench/canvas-engine/document/mergeVisible.ts',
  '/workbench/canvas-engine/engineStores.ts',
  '/workbench/canvas-engine/render/adjustments.ts',
  '/workbench/canvas-engine/render/alphaBounds.ts',
  '/workbench/canvas-engine/render/compositor.ts',
  '/workbench/canvas-engine/render/controlTransparency.ts',
  '/workbench/canvas-engine/render/exactAdjustments.ts',
  '/workbench/canvas-engine/render/groupCompositeScopes.ts',
  '/workbench/canvas-engine/render/maskFill.ts',
  '/workbench/canvas-engine/render/rasterComposite.ts',
  '/workbench/canvas-engine/samCoordinates.ts',
];

/** The public entry only re-exports the policy, so it rides in the policy's chunk; alone it costs a request. */
const CANVAS_PROCESSING_SIZE_MODULES = [
  '/features/generation/canvasProcessingSize.ts',
  '/features/generation/core/canvas/canvasProcessingSize.ts',
];

const matchesAnySuffix = (id: string, suffixes: readonly string[]) => suffixes.some((suffix) => id.endsWith(suffix));

const getLegacyChunkName = (id: string): string | null => {
  if (
    matchesAnySuffix(id, [
      '/platform/state/selectors.ts',
      '/platform/state/singleFlight.ts',
      '/workbench/palette/paletteStore.ts',
      '/platform/search/dateTokens.ts',
      '/platform/performance/semanticReady.ts',
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
      '/workbench/hotkeys/keyGlyphs.tsx',
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

  // Only a workflow thumbnail snapshot loads it, on demand.
  if (id.includes('/node_modules/html-to-image/')) {
    return 'html-to-image';
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
    // An env define, not a global: Vitest browser mode re-encodes string globals, wrapping them in quotes.
    'import.meta.env.APP_VERSION': JSON.stringify(readAppVersion()),
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
              test: (id) =>
                matchesAnySuffix(id, CANVAS_LAYER_SHARED_MODULES) ||
                (id.includes('/src/workbench/canvas-engine/') &&
                  !matchesAnySuffix(id, WORKBENCH_RUNTIME_SHARED_MODULES) &&
                  !matchesAnySuffix(id, CANVAS_ENGINE_BOOT_MODULES)),
            },
            {
              includeDependenciesRecursively: false,
              name: 'canvasProcessingSize',
              priority: 30,
              test: (id) => matchesAnySuffix(id, CANVAS_PROCESSING_SIZE_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'gallery-picker',
              priority: 30,
              test: (id) => matchesAnySuffix(id, GALLERY_PICKER_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'gallery-backend',
              priority: 30,
              test: (id) => matchesAnySuffix(id, GALLERY_BACKEND_MODULES),
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
              name: 'workbench-runtime-shared',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WORKBENCH_RUNTIME_SHARED_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'workflow-core',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WORKFLOW_CORE_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'workflow-queries',
              priority: 30,
              test: (id) => matchesAnySuffix(id, WORKFLOW_QUERIES_MODULES),
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
              name: 'image-actions',
              priority: 30,
              test: (id) => matchesAnySuffix(id, IMAGE_ACTIONS_MODULES),
            },
            {
              includeDependenciesRecursively: false,
              name: 'imageMapStore',
              priority: 30,
              test: (id) => matchesAnySuffix(id, IMAGE_MAP_DATA_MODULES),
            },
            {
              // Only the Image Map's model-missing message renders `<Trans>`; this keeps the HTML parser out of vendor.
              includeDependenciesRecursively: false,
              name: 'i18n-trans',
              priority: 30,
              test: (id) =>
                id.includes('/node_modules/html-parse-stringify/') ||
                /\/node_modules\/react-i18next\/dist\/es\/Trans(?:WithoutContext)?\.js$/.test(id),
            },
            {
              // Only the lazy command palette uses Chakra's Command parts. The `chakra` group below would pull them in
              // through the package barrel and load them at startup.
              includeDependenciesRecursively: false,
              name: 'chakra-command',
              priority: 30,
              test: (id) => /\/node_modules\/@chakra-ui\/react\/dist\/[^/]+\/components\/command\//.test(id),
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
      '@api': fileURLToPath(new URL('../api', import.meta.url)),
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
