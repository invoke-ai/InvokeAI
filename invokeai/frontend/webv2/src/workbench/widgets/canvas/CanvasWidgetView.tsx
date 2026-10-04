import type { WidgetViewProps } from '@workbench/widgetContracts';
import type { MouseEvent as ReactMouseEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import { useQueueItemProgressImage } from '@features/queue/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { preloadCanvasInvocation } from '@workbench/activeInvocationSubmission';
import { getCanvasImportNotice } from '@workbench/canvas-operations/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { recordCanvasImportError } from '@workbench/image-actions/canvasImportError';
import { readLayerPanelState } from '@workbench/layerPanelState';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useNotify } from '@workbench/useNotify';
import { CanvasLayerContextMenu } from '@workbench/widgets/layers/LayerContextMenu';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { lazy, Suspense, useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useModelGridSize } from './bboxGrid';
import { CanvasBottomOverlay } from './CanvasBottomOverlay';
import { copyBlobToClipboard, decodeImageBlob, readClipboardImage } from './canvasClipboard';
import {
  resolveCanvasContextMenu,
  resolveCanvasContextMenuBranch,
  type CanvasContextMenuTarget,
} from './canvasContextMenu';
import { CanvasCreateFromBboxSubmenu } from './CanvasCreateFromBboxSubmenu';
import { CanvasEditRefusalNotices } from './CanvasEditRefusalNotices';
import { CanvasGlobalContextMenu } from './CanvasGlobalContextMenu';
import { executeCanvasHotkeyCommand } from './canvasHotkeyCommands';
import { resolveCanvasImageDrop } from './canvasImageDnd';
import { CanvasImageDropOverlay } from './CanvasImageDropOverlay';
import { getCanvasInteractionCapabilities } from './canvasInteractionLock';
import { CanvasSaveToGallerySubmenu } from './CanvasSaveToGallerySubmenu';
import {
  CANVAS_SETTINGS,
  CANVAS_SHOW_PROGRESS_KEY,
  canvasSettingsEqual,
  resolveCanvasSettings,
} from './canvasSettings';
import { CanvasSurface } from './CanvasSurface';
import { CanvasSurfaceContextLayout } from './CanvasSurfaceContextLayout';
import { resolveCheckerColors } from './checkerColors';
import { CanvasColorFeed } from './color-system/CanvasColorFeed';
import { useActiveColorCommands } from './color-system/useActiveColors';
import { useCanvasOperation } from './engineStoreHooks';
import { executeCanvasImageDropImport } from './executeCanvasImageDropImport';
import { StagingBar } from './StagingBar';
import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';
import { INLINE_EDIT_SELECTOR } from './surfaceFocus';
import { ToolStrip } from './ToolStrip';
import { useCanvasGallerySave } from './useCanvasGallerySave';
import { useCreateFromBbox } from './useCreateFromBbox';
import { reportLayerOperation, reportPreparedCommit, reportStructuralCommit } from './useStructuralCommit';

const MissingFontsDialog = lazy(() =>
  import('./MissingFontsDialog').then((module) => ({ default: module.MissingFontsDialog }))
);

/**
 * Wire reducer chrome, commands, settings, and staging around the pixel/input engine. Properties owns tool
 * settings; {@link CanvasHeaderActions} owns view controls.
 */
export const CanvasWidgetView = ({ runtime }: WidgetViewProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { canvas: canvasCommands, notifications, queue } = useWorkbenchCommands();
  const canvasDispatch = useCanvasProjectMutationDispatch();
  const queries = useWorkbenchQueries();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const antialiasProgressImages = useActiveProjectSelector((project) => project.settings.antialiasProgressImages);
  const { document, stagingArea } = canvas;
  const fontReferences = useMemo(() => engine?.fonts.collectReferences(document) ?? [], [document, engine]);
  const operation = useCanvasOperation(engine);
  const operationKind = operation?.status === 'active' ? operation.identity.kind : null;
  const { isSaving, save: saveToGallery } = useCanvasGallerySave(engine);
  const { createFromBbox, isCreating } = useCreateFromBbox(engine);

  // Warm the split invocation chunk when Canvas mounts so the first submission need not download it.
  useMountEffect(preloadCanvasInvocation);

  // Right-click on the canvas surface targets the selected layer (the panel is
  // the sole authority; the stack is never hit-tested) and opens either the
  // shared per-layer menu or the global menu at the pointer.
  const [contextMenuTarget, setContextMenuTarget] = useState<CanvasContextMenuTarget | null>(null);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  // Feed model grid policy into the model-agnostic engine using primitive selectors for value-stable
  // subscriptions.
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
  const settings = useActiveProjectSelector(
    (project) => resolveCanvasSettings(getProjectWidgetValues(project, 'canvas')),
    canvasSettingsEqual
  );
  useEffect(() => {
    if (!engine) {
      return;
    }
    for (const setting of CANVAS_SETTINGS) {
      // Only engine-backed settings feed a store; settings consumed elsewhere
      // in the frontend have no store and are skipped here.
      if (setting.store) {
        engine.interaction.set(setting.store, settings[setting.key]);
      }
    }
  }, [engine, settings]);

  // The pair↔engine bridge lives in a null child so per-pointermove pair
  // edits never re-render this shell; the commands feed the X/D hotkeys below.
  const colorCommands = useActiveColorCommands();

  // Resolve checker colors from live theme tokens after ThemeController updates data-theme, then feed the engine
  // store.
  const themeId = useWorkbenchSettingsSelector((snapshot) => snapshot.preferences.themeId);
  useEffect(() => {
    engine?.interaction.set('checkerColors', resolveCheckerColors());
  }, [engine, themeId]);

  const stagingSlots = getCanvasStagingSlots(canvas, queueItems);
  const selectedSlot = stagingSlots[stagingArea.selectedImageIndex];
  const selectedCandidate = selectedSlot?.kind === 'candidate' ? selectedSlot.candidate : undefined;
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
  const interactionCapabilities = getCanvasInteractionCapabilities({
    hasCanvasEngine: engine !== null,
    hasSelectedCandidate: selectedCandidate !== undefined,
    hasStagingSlots,
    isCanvasGenerationInFlight,
    operationKind,
  });
  const isInteractionLocked = interactionCapabilities.isSurfaceInteractionLocked;
  const handleSurfaceContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      // Preserve native menus in inline editors. Canvas context actions target the selected layer without changing
      // selection.
      const resolution = resolveCanvasContextMenu({
        clientX: event.clientX,
        clientY: event.clientY,
        isInlineEditor: event.target instanceof Element && !!event.target.closest(INLINE_EDIT_SELECTOR),
        isInteractionLocked,
        selectedLayerId: engine?.tools.canTargetLayerFromContextMenu() ? canvas.document.selectedLayerId : null,
      });
      if (!resolution.preventDefault) {
        return;
      }
      event.preventDefault();
      setContextMenuTarget(resolution.target);
    },
    [canvas.document.selectedLayerId, engine, isInteractionLocked]
  );

  const handleCanvasImageDrop = useCallback(
    (event: DragEndEvent) => {
      const resolution = resolveCanvasImageDrop(event.active.data.current, event.over?.data.current);
      if (!resolution) {
        return;
      }

      // Capture the destination project and mounted engine before any network
      // work so a project switch cannot retarget this import mid-flight.
      const project = queries.getSnapshot().activeProject;
      const mountedEngine = engine;

      const execute = async (): Promise<void> => {
        try {
          const result = await executeCanvasImageDropImport({
            destination: resolution.destination,
            canvas: canvasCommands,
            engine: mountedEngine,
            queries,
            imageNames: resolution.imageNames,
            project,
          });
          const notice = getCanvasImportNotice(result);
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
  const copySelection = useEffectEvent((cut: boolean) => {
    const mountedEngine = engine;
    if (!mountedEngine) {
      return;
    }
    void (async () => {
      try {
        const blob = await mountedEngine.selection.exportSelectionBlob();
        if (!blob) {
          return;
        }
        await copyBlobToClipboard(blob);
        if (cut) {
          // Only after the write succeeds — a failed copy must not destroy pixels.
          mountedEngine.selection.eraseSelection();
        }
      } catch {
        notifications.add({ kind: 'error', title: t('widgets.canvas.clipboard.copyFailed') });
      }
    })();
  });

  /** Pastes an image off the system clipboard as a new layer over the bbox. */
  const pasteFromClipboard = useEffectEvent(() => {
    const mountedEngine = engine;
    if (!mountedEngine) {
      return;
    }
    void (async () => {
      const blob = await readClipboardImage();
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
  });

  useDndMonitor({ onDragEnd: handleCanvasImageDrop });

  useLayoutEffect(() => {
    engine?.tools.setInteractionLocked(isInteractionLocked);
    return () => engine?.tools.setInteractionLocked(false);
  }, [engine, isInteractionLocked]);

  /* eslint-disable react/preserve-manual-memoization -- imperative engine payload is mutable by design */
  const commitSelectedStagedImage = useCallback(
    (continueStaging: boolean) => {
      if (selectedSlot?.kind !== 'candidate' || !engine) {
        return;
      }
      const result = engine.layers.commitStagedImage({
        candidate: selectedSlot.candidate,
        continueStaging,
        selectedImageIndex: stagingArea.selectedImageIndex,
      });
      if (result.status !== 'committed' && result.status !== 'busy') {
        // A candidate that left staging is as stale as one that changed under the accept.
        reportLayerOperation(result.status === 'missing' ? 'stale' : result.status, notify.error, t);
      }
    },
    [engine, notify, selectedSlot, stagingArea.selectedImageIndex, t]
  );
  /* eslint-enable react/preserve-manual-memoization */
  const acceptStagedImage = useCallback(() => commitSelectedStagedImage(false), [commitSelectedStagedImage]);
  const saveStagedImageAndContinue = useCallback(() => commitSelectedStagedImage(true), [commitSelectedStagedImage]);
  const cancelQueueItem = useCallback((queueItemId: string) => queue.cancel(undefined, queueItemId), [queue]);
  const cycleStagedImage = useCallback(
    (direction: -1 | 1) => canvasDispatch({ direction, type: 'cycleStagedImage' }),
    [canvasDispatch]
  );
  const discardAllStagedImages = useCallback(
    () => canvasDispatch({ type: 'discardAllStagedImages' }),
    [canvasDispatch]
  );
  const discardSelectedStagedImage = useCallback(
    () => canvasDispatch({ type: 'discardSelectedStagedImage' }),
    [canvasDispatch]
  );
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

  // "Show progress on canvas" gates ONLY the selected placeholder's live denoise
  // frame; a selected finished candidate still previews (that's staging, not progress).
  const selectedPlaceholderProgressImage = useQueueItemProgressImage(
    selectedPlaceholder?.queueItemId ?? '',
    selectedPlaceholder?.itemIndex ?? 0
  );
  const progressImage = settings[CANVAS_SHOW_PROGRESS_KEY] ? selectedPlaceholderProgressImage : null;

  const previewSource = selectStagedPreviewSource({
    bboxHeight: document.bbox.height,
    bboxWidth: document.bbox.width,
    isGenerationInFlight: selectedPlaceholder !== null,
    isVisible: stagingArea.isVisible,
    progressImage,
    selectedImageName: selectedCandidate?.imageName ?? null,
    selectedPlacement: selectedCandidate?.placement ?? null,
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
  // Clear the preview when the widget (or engine) goes away, so an accepted /
  // discarded candidate never lingers over the canvas.
  useEffect(() => {
    return () => engine?.previews.setStagedPreview(null);
  }, [engine]);

  const executeCanvasHotkey = useEffectEvent((commandId: string) => {
    const selectedLayerIds = readLayerPanelState(projectId, document.selectedLayerId).selectedIds;
    executeCanvasHotkeyCommand(commandId, {
      copySelection,
      dispatch: canvasDispatch,
      document,
      engine,
      hasSelectedStagedCandidate: selectedCandidate !== undefined,
      hasStagingSlots,
      isInteractionLocked,
      notifyLayerDuplicateFailed: () =>
        notifications.add({ kind: 'error', title: t('widgets.layers.actions.copyFailed') }),
      pasteFromClipboard,
      reportLayerOperation: (refusal) => reportLayerOperation(refusal, notify.error, t),
      reportPreparedCommit: (outcome) => reportPreparedCommit(outcome, notify.error, t),
      reportStructuralCommit: (result) => reportStructuralCommit(result, notify.error, t),
      resetActiveColors: colorCommands.resetPair,
      selectedLayerIds,
      swapActiveColors: colorCommands.swapPair,
      t,
    });
  });

  useEffect(() => {
    const hotkeys = [
      // Staging keeps `alt+[` / `alt+]`; bare left/right are registered as layer nudges,
      // then intercepted above to cycle staging slots while any slot exists.
      ['canvas.prevEntity', t('widgets.canvas.commands.previousEntity'), ['alt+[']],
      ['canvas.nextEntity', t('widgets.canvas.commands.nextEntity'), ['alt+]']],
      ['canvas.deleteSelected', t('widgets.canvas.commands.deleteSelected'), ['delete', 'backspace']],
      ['canvas.resetSelected', t('widgets.canvas.commands.resetSelected'), ['shift+c']],
      ['canvas.undo', t('widgets.canvas.commands.undo'), ['mod+z']],
      ['canvas.redo', t('widgets.canvas.commands.redo'), ['mod+shift+z', 'mod+y']],
      // Exclude editable targets so tool letters and size brackets do not intercept typing.
      ['canvas.tool.view', t('widgets.canvas.commands.selectViewTool'), ['h']],
      ['canvas.tool.move', t('widgets.canvas.commands.selectMoveTool'), ['v']],
      ['canvas.transformSelected', t('widgets.canvas.commands.selectTransformTool'), ['mod+t']],
      ['canvas.tool.bbox', t('widgets.canvas.commands.selectBboxTool'), []],
      ['canvas.tool.brush', t('widgets.canvas.commands.selectBrushTool'), ['b']],
      ['canvas.tool.eraser', t('widgets.canvas.commands.selectEraserTool'), ['e']],
      ['canvas.tool.lasso', t('widgets.canvas.commands.selectLassoTool'), ['l']],
      ['canvas.tool.marquee', t('widgets.canvas.commands.selectMarqueeTool'), ['u']],
      ['canvas.toggleNonRasterLayers', t('widgets.canvas.commands.toggleNonRasterLayers'), ['shift+h']],
      ['canvas.copySelection', t('widgets.canvas.commands.copySelection'), ['mod+c']],
      ['canvas.cutSelection', t('widgets.canvas.commands.cutSelection'), ['mod+x']],
      ['canvas.pasteImage', t('widgets.canvas.commands.pasteImage'), ['mod+v']],
      ['canvas.tool.shape', t('widgets.canvas.commands.selectShapeTool'), ['r']],
      ['canvas.tool.text', t('widgets.canvas.commands.selectTextTool'), ['t']],
      ['canvas.tool.gradient', t('widgets.canvas.commands.selectGradientTool'), ['g']],
      // Selection: select all / deselect / invert (engine-owned transient selection).
      ['canvas.selectAll', t('widgets.canvas.commands.selectAll'), ['mod+a']],
      ['canvas.deselect', t('widgets.canvas.commands.deselect'), ['mod+d']],
      ['canvas.invertSelection', t('widgets.canvas.commands.invertSelection'), ['mod+shift+i']],
      ['canvas.brushSizeDown', t('widgets.canvas.commands.decreaseBrushSize'), ['[']],
      ['canvas.brushSizeUp', t('widgets.canvas.commands.increaseBrushSize'), [']']],
      // The active color pair: X swaps, D resets to black/white.
      ['canvas.toggleFillColor', t('widgets.canvas.commands.swapColors'), ['x']],
      ['canvas.setFillColorsToDefault', t('widgets.canvas.commands.resetColors'), ['d']],
      // Move the selected layer: arrows nudge 1px, shift+arrows 10px.
      ['canvas.nudgeLeft', t('widgets.canvas.commands.nudgeLeft'), ['arrowleft']],
      ['canvas.nudgeRight', t('widgets.canvas.commands.nudgeRight'), ['arrowright']],
      ['canvas.nudgeUp', t('widgets.canvas.commands.nudgeUp'), ['arrowup']],
      ['canvas.nudgeDown', t('widgets.canvas.commands.nudgeDown'), ['arrowdown']],
      ['canvas.nudgeLeftLarge', t('widgets.canvas.commands.nudgeLeftLarge'), ['shift+arrowleft']],
      ['canvas.nudgeRightLarge', t('widgets.canvas.commands.nudgeRightLarge'), ['shift+arrowright']],
      ['canvas.nudgeUpLarge', t('widgets.canvas.commands.nudgeUpLarge'), ['shift+arrowup']],
      ['canvas.nudgeDownLarge', t('widgets.canvas.commands.nudgeDownLarge'), ['shift+arrowdown']],
      // Layer management.
      ['canvas.duplicateLayer', t('widgets.canvas.commands.duplicateLayer'), ['mod+j']],
      ['canvas.groupLayers', t('widgets.canvas.commands.groupLayers'), ['mod+g']],
      ['canvas.ungroupLayers', t('widgets.canvas.commands.ungroupLayers'), ['mod+shift+g']],
      ['canvas.mergeDown', t('widgets.canvas.commands.mergeDown'), ['mod+e']],
      ['canvas.layerForward', t('widgets.canvas.commands.layerForward'), ['mod+]']],
      ['canvas.layerBackward', t('widgets.canvas.commands.layerBackward'), ['mod+[']],
      ['canvas.layerToFront', t('widgets.canvas.commands.layerToFront'), ['mod+shift+]']],
      ['canvas.layerToBack', t('widgets.canvas.commands.layerToBack'), ['mod+shift+[']],
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeCanvasHotkey(id), id, title }),
      runtime.hotkeys.register({ allowInEditable: false, commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

  const layerContextMenuTarget = useMemo(
    () =>
      contextMenuTarget?.layerId !== null && contextMenuTarget?.layerId !== undefined
        ? { layerId: contextMenuTarget.layerId, x: contextMenuTarget.x, y: contextMenuTarget.y }
        : null,
    [contextMenuTarget]
  );
  const contextMenuBranch = resolveCanvasContextMenuBranch(contextMenuTarget, engine !== null);
  // The two composite operations share one busy flag so they can't overlap.
  const isCompositeMenuDisabled = !engine || isSaving || isCreating || isInteractionLocked;
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
      </>
    ),
    [createFromBboxSubmenu, saveToGallerySubmenu]
  );
  const canvasSurface = useMemo(() => (engine ? <CanvasSurface engine={engine} /> : null), [engine]);

  return (
    <Box
      aria-label={t('widgets.canvas.surface')}
      bg="bg.inset"
      h="full"
      overflow="hidden"
      position="relative"
      role="region"
      w="full"
    >
      <CanvasColorFeed engine={engine} />
      {engine ? <CanvasEditRefusalNotices key={projectId} engine={engine} /> : null}
      {engine && fontReferences.length > 0 ? (
        <Suspense fallback={null}>
          <MissingFontsDialog key={projectId} engine={engine} groups={fontReferences} />
        </Suspense>
      ) : null}
      <CanvasSurfaceContextLayout surface={canvasSurface} onContextMenu={handleSurfaceContextMenu}>
        <CanvasImageDropOverlay
          isDocumentEditingLocked={interactionCapabilities.isDocumentEditingLocked}
          isInteractionLocked={isInteractionLocked}
        />
        {engine ? (
          <>
            <ToolStrip engine={engine} isInteractionLocked={isInteractionLocked} />
            <CanvasLayerContextMenu
              beforeDangerItems={compositeSubmenus}
              dispatch={canvasDispatch}
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
                canAccept={interactionCapabilities.canAcceptStagedImage}
                hasMultipleSlots={hasMultipleStagingSlots}
                isGenerating={isCanvasGenerationInFlight}
                isVisible={stagingArea.isVisible}
                selectedCandidate={selectedCandidate}
                selectedImageIndex={stagingArea.selectedImageIndex}
                selectedSlot={selectedSlot}
                slots={stagingSlots}
                onAccept={acceptStagedImage}
                onCancelQueueItem={cancelQueueItem}
                onCycle={cycleStagedImage}
                onDiscardAll={discardAllStagedImages}
                onDiscardSelected={discardSelectedStagedImage}
                onPreloadCandidate={preloadStagedCandidate}
                onSelectImage={selectStagedImage}
                onSaveToLayerAndContinue={saveStagedImageAndContinue}
                onSetAutoSwitch={setStagingAutoSwitch}
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
