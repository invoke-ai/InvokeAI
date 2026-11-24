import type { WidgetRuntimeApi, WidgetViewProps } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';
import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import { classifyGalleryUpload } from '@features/gallery/contracts';
import { useQueueItemProgressImage } from '@features/queue/react';
import { ClipboardUnavailableError, copyImage } from '@platform/browser/clipboard';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useNativeFileDrag } from '@platform/react/useNativeFileDrag';
import { useQueryClient } from '@tanstack/react-query';
import { preloadCanvasInvocation } from '@workbench/activeInvocationSubmission';
import { getDocumentNode } from '@workbench/canvas-engine/api';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { getCanvasStagingSlotRect, getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { useRegisterShortcutHintSource } from '@workbench/hotkeys/hintSources';
import { recordCanvasImportError } from '@workbench/image-actions/canvasImportError';
import { readLayerPanelState } from '@workbench/layerPanelState';
import { useWorkbenchPreferenceSelector, useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useNotify } from '@workbench/useNotify';
import { CanvasLayerContextMenu } from '@workbench/widgets/layers/LayerContextMenu';
import { useDroppedFileUpload } from '@workbench/widgets/shared/useDroppedFileUpload';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { lazy, Suspense, useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { VectorPathMenuTarget } from './VectorPathContextMenu';

import { useModelGridSize } from './bboxGrid';
import { CanvasBottomOverlay } from './CanvasBottomOverlay';
import { copyCanvasSelection, decodeImageBlob, readClipboardImage } from './canvasClipboard';
import {
  cropAllLayersToFrame,
  resolveCanvasContextMenu,
  resolveCanvasContextMenuBranch,
  type CanvasContextMenuTarget,
} from './canvasContextMenu';
import { CanvasCreateFromBboxSubmenu } from './CanvasCreateFromBboxSubmenu';
import { CanvasCropAllLayersMenuItem } from './CanvasCropAllLayersMenuItem';
import { CanvasEditRefusalNotices } from './CanvasEditRefusalNotices';
import { CanvasFrameLockFeed } from './CanvasFrameLockFeed';
import { CanvasFrameSizeReadout } from './CanvasFrameSizeReadout';
import { CanvasGlobalContextMenu } from './CanvasGlobalContextMenu';
import { CANVAS_ONLY_HOTKEYS, executeCanvasHotkeyCommand } from './canvasHotkeyCommands';
import { findCanvasImageDropDestination, resolveCanvasImageDrop, type CanvasImageDropData } from './canvasImageDnd';
import { CanvasImageDropOverlay } from './CanvasImageDropOverlay';
import { CanvasSaveToGallerySubmenu } from './CanvasSaveToGallerySubmenu';
import { CanvasScrollInputFeed } from './CanvasScrollInputFeed';
import {
  CANVAS_SETTINGS,
  CANVAS_SHOW_PROGRESS_KEY,
  canvasSettingsEqual,
  resolveCanvasSettings,
} from './canvasSettings';
import { CanvasStagedRegionsFeed } from './CanvasStagedRegionsFeed';
import { CanvasSurface } from './CanvasSurface';
import { CanvasSurfaceContextLayout } from './CanvasSurfaceContextLayout';
import {
  CANVAS_COMPANION_HOTKEYS,
  type HotkeyTableEntry,
  reportMergeDownRefusal,
  reportTransformRefusal,
  showCanvasRefusalNotice,
} from './canvasToolCommands';
import { resolveCheckerColors } from './checkerColors';
import { CanvasColorFeed } from './color-system/CanvasColorFeed';
import { useActiveColorCommands } from './color-system/useActiveColors';
import { ControlLayerCanvasOverlay } from './ControlLayerCanvasOverlay';
import { useCanvasOperation } from './engineStoreHooks';
import { executeCanvasImageDropImport } from './executeCanvasImageDropImport';
import { createCanvasShortcutHintSource } from './shortcutHints';
import { acceptStagedCandidate } from './stagedAcceptance';
import { StagingBar } from './StagingBar';
import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';
import { INLINE_EDIT_SELECTOR } from './surfaceFocus';
import { ToolStrip } from './ToolStrip';
import { useCanvasGallerySave } from './useCanvasGallerySave';
import { useStagingRecovery } from './useCanvasRecovery';
import { useCreateFromBbox } from './useCreateFromBbox';
import { useStagedResultGallerySave } from './useStagedResultGallerySave';
import { reportLayerOperation, reportPreparedCommit, reportStructuralCommit } from './useStructuralCommit';

const VectorPathContextMenu = lazy(() =>
  import('./VectorPathContextMenu').then((module) => ({ default: module.VectorPathContextMenu }))
);

const MissingFontsDialog = lazy(() =>
  import('./MissingFontsDialog').then((module) => ({ default: module.MissingFontsDialog }))
);

/** The shared engine stays stable for this keyed project/instance lease. */
const CanvasShortcutHintRegistration = ({
  engine,
  projectId,
  instanceId,
}: {
  engine: CanvasEngineHandle;
  projectId: string;
  instanceId: string;
}) => {
  const [source] = useState(() => createCanvasShortcutHintSource(engine, projectId, instanceId));
  useRegisterShortcutHintSource(source);
  return null;
};

// Reset only the registration lease when its APIs or translator change, leaving Canvas state mounted.
let hotkeyRegistrationKeys: WeakMap<object, number> | null = null;
let nextHotkeyRegistrationKey = 0;
const hotkeyRegistrationKey = (identity: object): number => {
  hotkeyRegistrationKeys ??= new WeakMap();
  let key = hotkeyRegistrationKeys.get(identity);
  if (key === undefined) {
    key = ++nextHotkeyRegistrationKey;
    hotkeyRegistrationKeys.set(identity, key);
  }
  return key;
};

const CanvasHotkeyRegistration = ({
  commands,
  hotkeys,
  execute,
  t,
}: Pick<WidgetRuntimeApi, 'commands' | 'hotkeys'> & { execute: (commandId: string) => void; t: TFunction }) => {
  const executeCommand = useEffectEvent(execute);
  // useMountEffect is the repository's mount/unmount effect boundary.
  /* eslint-disable react-hooks/rules-of-hooks */
  useMountEffect(() => {
    const entries: readonly HotkeyTableEntry[] = [...CANVAS_COMPANION_HOTKEYS, ...CANVAS_ONLY_HOTKEYS];
    const disposers = entries.flatMap(([id, titleKey, defaultKeys, options]) => {
      const title = t(titleKey);
      return [
        commands.register({ handler: () => executeCommand(id), id, title }),
        hotkeys.register({
          allowInEditable: false,
          commandId: id,
          defaultKeys: [...defaultKeys],
          id,
          title,
          ...options,
        }),
      ];
    });
    return () => disposers.forEach((dispose) => dispose());
  });
  /* eslint-enable react-hooks/rules-of-hooks */
  return null;
};

/** Wire reducer chrome, commands, settings, and staging around the pixel/input engine. */
export const CanvasWidgetView = ({ runtime, instance }: WidgetViewProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { canvas: canvasCommands, notifications, queue } = useWorkbenchCommands();
  const canvasDispatch = useCanvasProjectMutationDispatch();
  const queries = useWorkbenchQueries();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const antialiasProgressImages = useWorkbenchPreferenceSelector((preferences) => preferences.antialiasProgressImages);
  const { document, stagingArea } = canvas;
  const fontReferences = useMemo(() => engine?.fonts.collectReferences(document) ?? [], [document, engine]);
  const operation = useCanvasOperation(engine);
  const operationKind = operation?.status === 'active' ? operation.identity.kind : null;
  const { isSaving, save: saveToGallery } = useCanvasGallerySave(engine);
  const { createFromBbox, isCreating } = useCreateFromBbox(engine);
  const stagingRecovery = useStagingRecovery();

  // Warm the split invocation chunk when Canvas mounts so the first submission need not download it.
  useMountEffect(preloadCanvasInvocation);

  // Right-click on the canvas surface targets the selected layer (the panel is
  // the sole authority; the stack is never hit-tested) and opens either the
  // shared per-layer menu or the global menu at the pointer.
  const [vectorMenuTarget, setVectorMenuTarget] = useState<VectorPathMenuTarget | null>(null);
  const vectorMenuReturnFocus = useRef<HTMLElement | null>(null);
  const closeVectorMenu = useCallback(() => {
    setVectorMenuTarget(null);
    const focusTarget = vectorMenuReturnFocus.current;
    vectorMenuReturnFocus.current = null;
    const focused = focusTarget?.ownerDocument.activeElement;
    if (focusTarget?.isConnected && (focused === focusTarget.ownerDocument.body || focused?.closest('[role="menu"]'))) {
      focusTarget.focus({ preventScroll: true });
    }
  }, []);
  const [contextMenuTarget, setContextMenuTarget] = useState<CanvasContextMenuTarget | null>(null);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  // Feed model grid policy into the model-agnostic engine using primitive selectors for value-stable subscriptions.
  const modelBase = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate') as { model?: { base?: unknown } } | undefined;
    return typeof values?.model?.base === 'string' ? values.model.base : null;
  });
  const modelVariant = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate') as { model?: { variant?: unknown } } | undefined;
    return typeof values?.model?.variant === 'string' ? values.model.variant : null;
  });
  const bboxGrid = useModelGridSize(modelBase, modelVariant);
  useEffect(() => {
    engine?.viewport.setBboxGrid(bboxGrid);
  }, [bboxGrid, engine]);

  // Feed persisted per-project view settings into engine stores; header controls write the widget values.
  const settings = useWidgetValuesSelector('canvas', resolveCanvasSettings, canvasSettingsEqual);
  useEffect(() => {
    if (!engine) {
      return;
    }
    for (const setting of CANVAS_SETTINGS) {
      // Only engine-backed settings feed a store; settings without one are consumed elsewhere in the frontend.
      if (setting.store) {
        engine.interaction.set(setting.store, settings[setting.key]);
      }
    }
  }, [engine, settings]);

  // The pair↔engine bridge lives in a null child so per-pointermove pair edits never re-render this shell; the commands
  // feed the X/D hotkeys below.
  const colorCommands = useActiveColorCommands();

  // Resolve checker colors from live theme tokens after ThemeController updates data-theme, then feed the engine store.
  const themeId = useWorkbenchSettingsSelector((snapshot) => snapshot.preferences.themeId);
  useEffect(() => {
    engine?.interaction.set('checkerColors', resolveCheckerColors());
  }, [engine, themeId]);

  const stagingSlots = getCanvasStagingSlots(canvas, queueItems);
  const selectedSlot = stagingSlots[stagingArea.selectedImageIndex];
  const selectedCandidate = selectedSlot?.kind === 'candidate' ? selectedSlot.candidate : undefined;
  const { isSaving: isSavingStagedResult, save: saveStagedResult } = useStagedResultGallerySave(
    selectedCandidate?.imageName ?? null
  );
  const selectedPlaceholder = selectedSlot?.kind === 'placeholder' ? selectedSlot : null;
  const hasStagingSlots = stagingSlots.length > 0;
  const hasMultipleStagingSlots = stagingSlots.length > 1;
  const isCanvasGenerationInFlight = queueItems.some(
    (item) =>
      item.snapshot.destination === 'canvas' &&
      (item.status === 'pending' || item.status === 'running') &&
      // Show progress only for this documentRevision; wholesale session swaps invalidate earlier denoise frames.
      item.snapshot.canvas.documentRevision === canvas.documentRevision
  );
  // Generation and staged results never lock the canvas; only a running operation (Filter, Select Object) holds the
  // document, and the engine's own guards refuse what cannot run meanwhile.
  const isDocumentEditingLocked = operationKind !== null;
  const canAcceptStagedImage = engine !== null && selectedCandidate !== undefined && !isDocumentEditingLocked;

  // Files dragged in from the desktop show the gallery drag's zones and land the same way once uploaded.
  const queryClient = useQueryClient();
  const startDroppedFileUpload = useDroppedFileUpload({
    pendingKey: 'widgets.canvas.fileDrop.pending',
    surface: 'canvas',
    uploadingKey: 'widgets.canvas.fileDrop.uploading',
  });
  const importDroppedFiles = useCallback(
    (files: File[], destination: CanvasImageDropData['destination']) => {
      const images = files.filter((file) => classifyGalleryUpload(file)?.kind === 'image');
      if (images.length < files.length) {
        notifications.add({
          kind: images.length === 0 ? 'error' : 'info',
          title: t('widgets.canvas.fileDrop.notImage'),
        });
      }
      if (images.length === 0) {
        return;
      }
      startDroppedFileUpload(images, {
        load: () => import('./importCanvasFileDrop'),
        run: ({ importCanvasFileDrop }, context) =>
          importCanvasFileDrop({
            ...context,
            canvas: canvasCommands,
            destination,
            files: images,
            notifications,
            queries,
            queryClient,
            t,
          }),
        runFailedMessage: t('widgets.canvas.import.failed'),
      });
    },
    [canvasCommands, notifications, queries, queryClient, startDroppedFileUpload, t]
  );
  const openSurfaceContextMenu = useCallback(
    (surface: HTMLDivElement, clientX: number, clientY: number, keyboard: boolean) => {
      setVectorMenuTarget(null);
      vectorMenuReturnFocus.current = null;
      const bounds = surface.getBoundingClientRect();
      const x = keyboard ? bounds.left + bounds.width / 2 : clientX;
      const y = keyboard ? bounds.top + bounds.height / 2 : clientY;
      // Keyboard invocation acts on the current path; only a pointer invocation retargets by geometry.
      if (keyboard || engine?.layers.vectorEdit.targetContextMenu({ x: x - bounds.left, y: y - bounds.top })) {
        const edit = engine?.interaction.get('vectorEditSession');
        if (edit?.activePathId) {
          const focused = surface.ownerDocument.activeElement;
          if (keyboard && focused instanceof HTMLElement && surface.contains(focused)) {
            vectorMenuReturnFocus.current = focused;
          }
          setContextMenuTarget(null);
          setVectorMenuTarget({ x, y, layerId: edit.layerId, pathId: edit.activePathId });
          return;
        }
      }
      setContextMenuTarget(
        resolveCanvasContextMenu({
          clientX: x,
          clientY: y,
          isInlineEditor: false,
          selectedLayerId: engine?.tools.canTargetLayerFromContextMenu() ? canvas.document.selectedLayerId : null,
        }).target
      );
    },
    [canvas.document.selectedLayerId, engine]
  );
  const { drag: fileDrag, handlers: fileDragHandlers } = useNativeFileDrag({
    findZone: findCanvasImageDropDestination,
    onDrop: importDroppedFiles,
  });
  const handleSurfaceContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (event.target instanceof Element && event.target.closest(INLINE_EDIT_SELECTOR)) {
        return;
      }
      event.preventDefault();
      openSurfaceContextMenu(event.currentTarget, event.clientX, event.clientY, event.button <= 0);
    },
    [openSurfaceContextMenu]
  );
  const handleSurfaceMenuKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (
        event.defaultPrevented ||
        (event.key !== 'ContextMenu' &&
          !(event.key === 'F10' && event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey)) ||
        (event.target instanceof Element && event.target.closest(INLINE_EDIT_SELECTOR))
      ) {
        return;
      }
      event.preventDefault();
      openSurfaceContextMenu(event.currentTarget, 0, 0, true);
    },
    [openSurfaceContextMenu]
  );

  const handleCanvasImageDrop = useCallback(
    (event: DragEndEvent) => {
      const resolution = resolveCanvasImageDrop(event.active.data.current, event.over?.data.current);
      if (!resolution) {
        return;
      }

      // Capture the destination project and mounted engine before any network work so a project switch cannot retarget
      // this import mid-flight.
      const project = queries.getSnapshot().activeProject;
      const mountedEngine = engine;

      const execute = async (): Promise<void> => {
        try {
          const notice = await executeCanvasImageDropImport({
            destination: resolution.destination,
            canvas: canvasCommands,
            engine: mountedEngine,
            queries,
            imageNames: resolution.imageNames,
            project,
          });
          notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });
        } catch (error: unknown) {
          recordCanvasImportError({
            error,
            localizedMessage: t('widgets.canvas.import.failed'),
            notifications,
            projectId: project.id,
          });
        }
      };

      void execute();
    },
    [canvasCommands, engine, notifications, queries, t]
  );

  /** The engine produces selected pixels; widget code owns system clipboard writes and optional cutting. */
  const copySelection = useCallback(
    (cut: boolean) => {
      const mountedEngine = engine;
      if (!mountedEngine) {
        return;
      }
      void (async () => {
        try {
          await copyCanvasSelection(mountedEngine.selection, cut, copyImage);
        } catch (error) {
          notifications.add({
            kind: 'error',
            message: error instanceof ClipboardUnavailableError ? t('common.copyUnavailable') : undefined,
            title: t('widgets.canvas.clipboard.copyFailed'),
          });
        }
      })();
    },
    [engine, notifications, t]
  );

  /** Pastes an image off the system clipboard as a new layer over the bbox. */
  const pasteFromClipboard = useCallback(() => {
    const mountedEngine = engine;
    if (!mountedEngine) {
      return;
    }
    void (async () => {
      let blob: Blob | null;
      try {
        blob = await readClipboardImage();
      } catch (error) {
        // A missing clipboard API (plain HTTP) gets its reason; an item that fails to yield its image does not.
        notifications.add({
          kind: 'error',
          message: error instanceof ClipboardUnavailableError ? t('common.pasteUnavailable') : undefined,
          title: t('widgets.canvas.clipboard.pasteFailed'),
        });
        return;
      }
      if (!blob) {
        return;
      }
      const pixels = await decodeImageBlob(blob);
      if (!pixels) {
        notifications.add({ kind: 'error', title: t('widgets.canvas.clipboard.pasteFailed') });
        return;
      }
      const result = mountedEngine.selection.pasteImage(pixels);
      if (result.status !== 'created') {
        reportLayerOperation(result.status, notify.error, t);
      }
    })();
  }, [engine, notifications, notify.error, t]);

  useDndMonitor({ onDragEnd: handleCanvasImageDrop });

  /* eslint-disable react/preserve-manual-memoization -- imperative engine payload is mutable by design */
  const commitSelectedStagedImage = useCallback(
    (continueStaging: boolean) => {
      if (selectedSlot?.kind !== 'candidate' || !engine) {
        return;
      }
      const before = { documentRevision: canvas.documentRevision, pendingImages: stagingArea.pendingImages };
      const setsAsideOthers = !continueStaging && stagingArea.pendingImages.length > 1;
      let stoppedBatch = false;
      const result = acceptStagedCandidate(
        {
          commit: (options) => engine.layers.commitStagedImage(options),
          // With results set aside, the accept's own notice says the batch stopped, beside its Undo.
          stopBatch: (queueItemId) => {
            stoppedBatch = true;
            queue.cancel(
              projectId,
              queueItemId,
              setsAsideOthers
                ? null
                : {
                    message: t('widgets.canvas.staging.acceptStopMessage'),
                    title: t('widgets.canvas.staging.acceptStopTitle'),
                  }
            );
          },
        },
        {
          candidate: selectedSlot.candidate,
          continueStaging,
          queueItems,
          selectedImageIndex: stagingArea.selectedImageIndex,
        }
      );
      if (result.status === 'committed') {
        if (setsAsideOthers) {
          stagingRecovery.reportAccepted(before, selectedSlot.candidate, stoppedBatch);
        }
      } else if (result.status !== 'busy') {
        // A candidate that left staging is as stale as one that changed under the accept.
        reportLayerOperation(result.status === 'missing' ? 'stale' : result.status, notify.error, t);
      }
    },
    [
      canvas.documentRevision,
      engine,
      notify,
      projectId,
      queue,
      queueItems,
      selectedSlot,
      stagingArea.pendingImages,
      stagingArea.selectedImageIndex,
      stagingRecovery,
      t,
    ]
  );
  /* eslint-enable react/preserve-manual-memoization */
  const acceptStagedImage = useCallback(() => commitSelectedStagedImage(false), [commitSelectedStagedImage]);
  const saveStagedImageAndContinue = useCallback(() => commitSelectedStagedImage(true), [commitSelectedStagedImage]);
  const cycleStagedImage = useCallback(
    (direction: -1 | 1) => canvasDispatch({ direction, type: 'cycleStagedImage' }),
    [canvasDispatch]
  );
  const discardAllStagedImages = stagingRecovery.discardAll;
  const discardSelectedStagedImage = stagingRecovery.discardSelected;
  const preloadStagedCandidate = useCallback(
    (imageName: string) => engine?.previews.preloadStagedPreview(imageName),
    [engine]
  );
  const selectStagedImage = useCallback(
    (imageIndex: number) => canvasDispatch({ imageIndex, type: 'setStagedImageIndex' }),
    [canvasDispatch]
  );
  const setStagingAutoSwitch = useCallback(
    (mode: 'off' | 'latest' | 'progress') => canvasDispatch({ mode, type: 'setCanvasStagingAutoSwitch' }),
    [canvasDispatch]
  );
  const toggleStagingThumbnails = useCallback(
    () => canvasDispatch({ type: 'toggleCanvasStagingThumbnailsVisibility' }),
    [canvasDispatch]
  );
  const toggleStagingVisibility = useCallback(
    () => canvasDispatch({ type: 'toggleCanvasStagingVisibility' }),
    [canvasDispatch]
  );

  // "Show progress on canvas" gates ONLY the selected placeholder's live denoise frame; a selected finished candidate
  // still previews (that's staging, not progress).
  const selectedPlaceholderProgressImage = useQueueItemProgressImage(
    selectedPlaceholder?.queueItemId ?? '',
    selectedPlaceholder?.itemIndex ?? 0
  );
  const progressImage = settings[CANVAS_SHOW_PROGRESS_KEY] ? selectedPlaceholderProgressImage : null;

  const previewSource = selectStagedPreviewSource({
    isVisible: stagingArea.isVisible,
    progressImage,
    progressRect: selectedPlaceholder ? getCanvasStagingSlotRect(selectedPlaceholder) : null,
    selectedCandidate: selectedCandidate ?? null,
  });
  const previewKey = stagedPreviewKey(previewSource);

  // Decode staged previews only when previewKey changes, including progress frames; read current source without
  // retriggering on unrelated renders.
  const applyStagedPreview = useEffectEvent(() => {
    engine?.previews.setStagedPreview(previewSource);
  });
  useEffect(() => {
    applyStagedPreview();
  }, [engine, previewKey]);
  // Clear the preview when the widget (or engine) goes away, so an accepted / discarded candidate never lingers over
  // the canvas.
  useEffect(() => {
    return () => engine?.previews.setStagedPreview(null);
  }, [engine]);

  const executeCanvasHotkey = useCallback(
    (commandId: string) => {
      const selectedLayerIds = readLayerPanelState(projectId, document.selectedLayerId).selectedIds;
      executeCanvasHotkeyCommand(commandId, {
        copySelection,
        cropAllLayers: () => {
          if (engine) {
            void cropAllLayersToFrame(engine.layers, notify, t);
          }
        },
        dispatch: canvasDispatch,
        document,
        engine,
        hasMultipleStagingSlots,
        layerCommands: { engine, reportError: notify.error, t },
        pasteFromClipboard,
        reportLayerOperation: (refusal) => reportLayerOperation(refusal, notify.error, t),
        reportMergeDownRefusal: (refusal) => reportMergeDownRefusal(document, refusal, showCanvasRefusalNotice, t),
        reportTransformRefusal: (refusal, layerId) =>
          reportTransformRefusal(
            refusal,
            { id: layerId, name: getDocumentNode(document, layerId)?.name ?? '' },
            { error: notify.error, notice: showCanvasRefusalNotice },
            t
          ),
        reportPreparedCommit: (outcome) => reportPreparedCommit(outcome, notify.error, t),
        reportStructuralCommit: (result) => reportStructuralCommit(result, notify.error, t),
        resetActiveColors: colorCommands.resetPair,
        selectedLayerIds,
        swapActiveColors: colorCommands.swapPair,
        t,
      });
    },
    [
      canvasDispatch,
      colorCommands,
      copySelection,
      document,
      engine,
      hasMultipleStagingSlots,
      notify,
      pasteFromClipboard,
      projectId,
      t,
    ]
  );

  const layerContextMenuTarget = useMemo(
    () =>
      contextMenuTarget?.layerId !== null && contextMenuTarget?.layerId !== undefined
        ? { layerId: contextMenuTarget.layerId, x: contextMenuTarget.x, y: contextMenuTarget.y }
        : null,
    [contextMenuTarget]
  );
  const contextMenuBranch = resolveCanvasContextMenuBranch(contextMenuTarget, engine !== null);
  // The two composite operations share one busy flag so they can't overlap.
  const isCompositeMenuDisabled = !engine || isSaving || isCreating;
  const saveToGallerySubmenu = useMemo(
    () => <CanvasSaveToGallerySubmenu disabled={isCompositeMenuDisabled} onSave={saveToGallery} />,
    [isCompositeMenuDisabled, saveToGallery]
  );
  const createFromBboxSubmenu = useMemo(
    () => <CanvasCreateFromBboxSubmenu disabled={isCompositeMenuDisabled} onCreate={createFromBbox} />,
    [createFromBbox, isCompositeMenuDisabled]
  );
  const compositeSubmenus = useMemo(
    () => (
      <>
        {saveToGallerySubmenu}
        {createFromBboxSubmenu}
        {engine ? (
          <CanvasCropAllLayersMenuItem documentEditingLocked={isDocumentEditingLocked} layers={engine.layers} />
        ) : null}
      </>
    ),
    [createFromBboxSubmenu, engine, isDocumentEditingLocked, saveToGallerySubmenu]
  );
  const canvasSurface = useMemo(
    () => (engine ? <CanvasSurface engine={engine} instanceId={instance.id} /> : null),
    [engine, instance.id]
  );

  return (
    <Box
      aria-label={t('widgets.canvas.surface')}
      bg="bg.inset"
      h="full"
      overflow="hidden"
      position="relative"
      role="region"
      w="full"
      {...fileDragHandlers}
    >
      <CanvasHotkeyRegistration
        key={[runtime.commands, runtime.hotkeys, t].map(hotkeyRegistrationKey).join(':')}
        commands={runtime.commands}
        hotkeys={runtime.hotkeys}
        execute={executeCanvasHotkey}
        t={t}
      />
      {engine ? (
        <CanvasShortcutHintRegistration
          key={`${projectId}:${instance.id}`}
          engine={engine}
          projectId={projectId}
          instanceId={instance.id}
        />
      ) : null}
      <CanvasColorFeed engine={engine} />
      {engine ? <CanvasFrameLockFeed key={`${projectId}:frame-lock`} engine={engine} projectId={projectId} /> : null}
      {engine ? (
        <CanvasScrollInputFeed key={`${projectId}:scroll-input`} engine={engine} projectId={projectId} />
      ) : null}
      {engine ? <CanvasEditRefusalNotices key={`${projectId}:edit-refusals`} engine={engine} /> : null}
      {engine ? (
        <CanvasStagedRegionsFeed key={`${projectId}:staged-regions`} engine={engine} projectId={projectId} />
      ) : null}
      {engine && fontReferences.length > 0 ? (
        <Suspense fallback={null}>
          <MissingFontsDialog key={projectId} engine={engine} groups={fontReferences} />
        </Suspense>
      ) : null}
      <CanvasSurfaceContextLayout
        surface={canvasSurface}
        onContextMenu={handleSurfaceContextMenu}
        onKeyDown={handleSurfaceMenuKeyDown}
      >
        {!engine ? (
          <Box role="status" position="absolute" inset="0" display="grid" placeItems="center" color="fg.muted">
            {t('common.loading')}
          </Box>
        ) : null}
        <CanvasImageDropOverlay fileDrag={fileDrag} isDocumentEditingLocked={isDocumentEditingLocked} />
        {engine ? (
          <>
            <ToolStrip engine={engine} />
            {vectorMenuTarget ? (
              <Suspense fallback={null}>
                <VectorPathContextMenu engine={engine} target={vectorMenuTarget} onClose={closeVectorMenu} />
              </Suspense>
            ) : null}
            <CanvasFrameSizeReadout engine={engine} />
            <ControlLayerCanvasOverlay engine={engine} isFileDragActive={fileDrag !== null} />
            <CanvasLayerContextMenu
              beforeDangerItems={compositeSubmenus}
              engine={engine}
              showGroupLabels
              target={layerContextMenuTarget}
              onClose={closeContextMenu}
            />
          </>
        ) : null}
        {contextMenuBranch === 'global' && contextMenuTarget ? (
          <CanvasGlobalContextMenu target={contextMenuTarget} onClose={closeContextMenu}>
            {compositeSubmenus}
          </CanvasGlobalContextMenu>
        ) : null}

        {/* Staging keeps its bottom-center slot; the wrapper is click-through and the bar re-enables pointer events. */}
        <CanvasBottomOverlay.Root>
          {hasStagingSlots || isCanvasGenerationInFlight ? (
            <CanvasBottomOverlay.Staging>
              <StagingBar
                antialiasProgressImages={antialiasProgressImages}
                areThumbnailsVisible={stagingArea.areThumbnailsVisible}
                autoSwitchMode={stagingArea.autoSwitchMode}
                canAccept={canAcceptStagedImage}
                hasMultipleSlots={hasMultipleStagingSlots}
                isGenerating={isCanvasGenerationInFlight}
                isSavingToGallery={isSavingStagedResult}
                isVisible={stagingArea.isVisible}
                selectedCandidate={selectedCandidate}
                selectedImageIndex={stagingArea.selectedImageIndex}
                selectedSlot={selectedSlot}
                slots={stagingSlots}
                onAccept={acceptStagedImage}
                onCycle={cycleStagedImage}
                onDiscardAll={discardAllStagedImages}
                onDiscardSelected={discardSelectedStagedImage}
                onPreloadCandidate={preloadStagedCandidate}
                onSelectImage={selectStagedImage}
                onSaveToGallery={saveStagedResult}
                onSaveToLayerAndContinue={saveStagedImageAndContinue}
                onSetAutoSwitch={setStagingAutoSwitch}
                onStopGenerating={stagingRecovery.stopGenerating}
                onToggleThumbnails={toggleStagingThumbnails}
                onToggleVisibility={toggleStagingVisibility}
              />
            </CanvasBottomOverlay.Staging>
          ) : null}
        </CanvasBottomOverlay.Root>
      </CanvasSurfaceContextLayout>
    </Box>
  );
};
