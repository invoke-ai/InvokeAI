import type { WidgetViewProps } from '@workbench/widgetContracts';
import type { MouseEvent as ReactMouseEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import { useQueueItemProgressImage } from '@features/queue/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { preloadCanvasInvocation } from '@workbench/activeInvocationSubmission';
import { getCanvasImportNotice } from '@workbench/canvas-operations/api';
import { getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { recordCanvasImportError } from '@workbench/image-actions/canvasImportError';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { CanvasLayerContextMenu } from '@workbench/widgets/layers/LayerContextMenu';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { readLayerPanelSelection } from '@workbench/workbenchStore';
import { useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { gridSizeForModelBase } from './bboxGrid';
import { CanvasBottomControls } from './CanvasBottomControls';
import { CanvasBottomOverlay } from './CanvasBottomOverlay';
import { copyBlobToClipboard, decodeImageBlob, readClipboardImage } from './canvasClipboard';
import {
  resolveCanvasContextMenu,
  resolveCanvasContextMenuBranch,
  type CanvasContextMenuTarget,
} from './canvasContextMenu';
import { CanvasCreateFromBboxSubmenu } from './CanvasCreateFromBboxSubmenu';
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
import { useCanvasOperation } from './engineStoreHooks';
import { executeCanvasImageDropImport } from './executeCanvasImageDropImport';
import { StagingBar } from './StagingBar';
import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';
import { INLINE_EDIT_SELECTOR } from './surfaceFocus';
import { ToolStrip } from './ToolStrip';
import { useCanvasEngine } from './useCanvasEngine';
import { useCanvasGallerySave } from './useCanvasGallerySave';
import { useCreateFromBbox } from './useCreateFromBbox';

/**
 * The canvas widget shell. The engine owns pixels and interaction and renders
 * into {@link CanvasSurface}; this component only wires the reducer-backed
 * chrome around it — command/hotkey registration, the settings-store feed, and
 * the floating bottom chrome (tool options + staging). Zoom / fit / settings
 * live in the widget header ({@link CanvasHeaderActions}).
 */
export const CanvasWidgetView = ({ runtime }: WidgetViewProps) => {
  const { t } = useTranslation();
  const { canvas: canvasCommands, notifications, queue } = useWorkbenchCommands();
  const canvasDispatch = useCanvasProjectMutationDispatch();
  const queries = useWorkbenchQueries();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const antialiasProgressImages = useActiveProjectSelector((project) => project.settings.antialiasProgressImages);
  const { document, stagingArea } = canvas;
  const operation = useCanvasOperation(engine);
  const { isSaving, save: saveToGallery } = useCanvasGallerySave(engine);
  const { createFromBbox, isCreating } = useCreateFromBbox(engine);

  // Canvas invocation stays code-split from the rest of the workbench, but the
  // canvas being mounted is a strong intent signal. Warm it while the user edits
  // instead of making the first Ctrl+Enter pay the chunk download/evaluation.
  useMountEffect(preloadCanvasInvocation);

  // Right-click on the canvas surface: hit-test the layer under the cursor and
  // open either the shared per-layer menu or the global empty-space menu at the
  // pointer. Locked interaction skips the hit-test but keeps global save visible.
  const [contextMenuTarget, setContextMenuTarget] = useState<CanvasContextMenuTarget | null>(null);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  // The bbox tool snaps to a model-dependent grid; the engine is model-agnostic,
  // so read the active generate model's base and feed the grid size in.
  const modelBase = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate') as { model?: { base?: unknown } } | undefined;
    return typeof values?.model?.base === 'string' ? values.model.base : null;
  });
  useEffect(() => {
    engine?.viewport.setBboxGrid(gridSizeForModelBase(modelBase));
  }, [engine, modelBase]);

  // Canvas view settings (checkerboard / grid / invert-scroll) persist in the
  // canvas widget's per-project values; the engine only reads its stores, so
  // push the resolved values down whenever they change — same one-directional
  // feed as the bbox grid above. The header settings menu writes the values.
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

  // The checkerboard fills the whole (unbounded) canvas, so its two square colors
  // come from theme tokens rather than hardcoded greys. Resolve them from the live
  // Chakra theme and feed them into the engine's checker-colors store; re-resolve
  // whenever the theme (and thus color mode) changes. `themeId` flips
  // `<html data-theme>` in ThemeController's layout effect, which runs before this
  // passive effect in the same commit, so getComputedStyle reads the new theme.
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
      // Only this canvas SESSION's in-flight items: an item submitted before a
      // wholesale swap (new canvas / snapshot restore) belongs to a document
      // that no longer exists, so its denoise frames must not leak onto the
      // fresh canvas (F2). `documentRevision` bumps only on those swaps.
      item.snapshot.canvas.documentRevision === canvas.documentRevision
  );
  const interactionCapabilities = getCanvasInteractionCapabilities({
    hasCanvasEngine: engine !== null,
    hasSelectedCandidate: selectedCandidate !== undefined,
    hasStagingSlots,
    isCanvasGenerationInFlight,
    operationKind: operation?.status === 'active' ? operation.identity.kind : null,
  });
  const isInteractionLocked = interactionCapabilities.isSurfaceInteractionLocked;
  const handleSurfaceContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      // Keep the native menu inside inline editors (the text tool's contenteditable
      // overlay), consistent with the surface-focus INLINE_EDIT_SELECTOR.
      //
      // The menu targets the SELECTED layer, not the layer under the pointer, and
      // never dispatches a selection — the layers panel is the sole authority on
      // which layer is active.
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

  /**
   * Copies the selection's pixels to the system clipboard, optionally cutting
   * them. The engine produces the blob; the clipboard write is a widget concern
   * (`canvas-engine` may not reach `workbench/widgets`).
   */
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
        notifications.add({ kind: 'error', title: t('widgets.canvas.clipboard.pasteFailed') });
      }
    })();
  });

  useDndMonitor({ onDragEnd: handleCanvasImageDrop });

  useLayoutEffect(() => {
    engine?.tools.setInteractionLocked(isInteractionLocked);
    return () => engine?.tools.setInteractionLocked(false);
  }, [engine, isInteractionLocked]);

  /* eslint-disable react/react-compiler -- imperative engine payload is mutable by design */
  const commitSelectedStagedImage = useCallback(
    (continueStaging: boolean) => {
      if (selectedSlot?.kind === 'candidate') {
        engine?.layers.commitStagedImage({
          candidate: selectedSlot.candidate,
          continueStaging,
          selectedImageIndex: stagingArea.selectedImageIndex,
        });
      }
    },
    [engine, selectedSlot, stagingArea.selectedImageIndex]
  );
  const acceptStagedImage = useCallback(() => commitSelectedStagedImage(false), [commitSelectedStagedImage]);
  const saveStagedImageAndContinue = useCallback(() => commitSelectedStagedImage(true), [commitSelectedStagedImage]);
  /* eslint-enable react/react-compiler */
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

  // What the engine should draw as the staged preview: the live denoise-progress
  // frame while generating, else the selected candidate, else nothing. The pure
  // helper is unit-tested; the effect below drives the engine imperatively.
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

  // Syncing an external imperative system (the engine's staged preview) with
  // derived reducer/progress state is a genuine effect. `useEffectEvent` reads
  // the latest source without making it a dependency, so the decoding
  // `setStagedPreview` re-runs only when `previewKey` actually changes (which
  // includes every new progress frame) — never on unrelated re-renders.
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
    const selectedLayerIds = readLayerPanelSelection(projectId, document.selectedLayerId).selectedIds;
    executeCanvasHotkeyCommand(commandId, {
      copySelection,
      dispatch: canvasDispatch,
      document,
      engine,
      hasSelectedStagedCandidate: selectedCandidate !== undefined,
      hasStagingSlots,
      isInteractionLocked,
      pasteFromClipboard,
      selectedLayerIds,
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
      // Tool selection and brush/eraser size step. `allowInEditable: false` below
      // keeps these single-letter/bracket keys from firing while the user is
      // typing in a prompt/text field elsewhere in the workbench.
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
              layers={document.layers}
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

        {/*
         * Floating bottom-center chrome: the staging bar (when active) stacks
         * directly above the always-present tool options bar — "just like the
         * staging UI". The wrapper is click-through so the canvas stays
         * interactive around the bars; each bar re-enables pointer events.
         */}
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
          <CanvasBottomOverlay.Controls>
            <CanvasBottomControls
              engine={engine}
              isExternalInteractionLocked={isInteractionLocked}
              operation={operation}
            />
          </CanvasBottomOverlay.Controls>
        </CanvasBottomOverlay.Root>
      </CanvasSurfaceContextLayout>
    </Box>
  );
};
