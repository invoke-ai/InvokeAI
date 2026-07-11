/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { WidgetViewProps } from '@workbench/types';
import type { MouseEvent as ReactMouseEvent } from 'react';

import { Box, Stack } from '@chakra-ui/react';
import { useQueueItemProgressImage } from '@workbench/backend/progressImageStore';
import {
  createLayerId,
  deleteLayerActions,
  duplicateLayerActions,
  type LayerReorderKind,
  reorderIdsForHotkey,
  reorderLayerActions,
} from '@workbench/canvasLayerOps';
import { getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { CanvasLayerContextMenu, type CanvasLayerContextMenuTarget } from '@workbench/widgets/layers/LayerContextMenu';
import { canMergeLayerDown } from '@workbench/widgets/layers/layerOps';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useEffectEvent, useLayoutEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { gridSizeForModelBase } from './bboxGrid';
import { getCanvasInteractionCapabilities } from './canvasInteractionLock';
import {
  CANVAS_SETTINGS,
  CANVAS_SHOW_PROGRESS_KEY,
  canvasSettingsEqual,
  resolveCanvasSettings,
} from './canvasSettings';
import { CanvasSurface } from './CanvasSurface';
import { resolveCheckerColors } from './checkerColors';
import { useCanvasOperation } from './engineStoreHooks';
import { StagingBar } from './StagingBar';
import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';
import { INLINE_EDIT_SELECTOR } from './surfaceFocus';
import { CanvasOperationBar } from './tool-options/CanvasOperationBar';
import { ToolOptionsBar } from './tool-options/ToolOptionsBar';
import { ToolStrip } from './ToolStrip';
import { useCanvasEngine } from './useCanvasEngine';

/** Command id → document-space nudge delta (shift variants are ×10). */
const NUDGE_DELTAS: Record<string, { dx: number; dy: number }> = {
  'canvas.nudgeDown': { dx: 0, dy: 1 },
  'canvas.nudgeDownLarge': { dx: 0, dy: 10 },
  'canvas.nudgeLeft': { dx: -1, dy: 0 },
  'canvas.nudgeLeftLarge': { dx: -10, dy: 0 },
  'canvas.nudgeRight': { dx: 1, dy: 0 },
  'canvas.nudgeRightLarge': { dx: 10, dy: 0 },
  'canvas.nudgeUp': { dx: 0, dy: -1 },
  'canvas.nudgeUpLarge': { dx: 0, dy: -10 },
};

/** Command id → z-reorder direction (index 0 = top-most; "forward" moves toward 0). */
const REORDER_KINDS: Record<string, LayerReorderKind> = {
  'canvas.layerBackward': 'backward',
  'canvas.layerForward': 'forward',
  'canvas.layerToBack': 'back',
  'canvas.layerToFront': 'front',
};

/**
 * The canvas widget shell. The engine owns pixels and interaction and renders
 * into {@link CanvasSurface}; this component only wires the reducer-backed
 * chrome around it — command/hotkey registration, the settings-store feed, and
 * the floating bottom chrome (tool options + staging). Zoom / fit / settings
 * live in the widget header ({@link CanvasHeaderActions}).
 */
export const CanvasWidgetView = ({ runtime }: WidgetViewProps) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const engine = useCanvasEngine();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const antialiasProgressImages = useActiveProjectSelector((project) => project.settings.antialiasProgressImages);
  const { document, stagingArea } = canvas;
  const operation = useCanvasOperation(engine);

  // Right-click on the canvas surface: hit-test the layer under the cursor, select
  // it, and open the SAME per-layer context menu the layers panel uses — anchored
  // at the pointer. `null` (empty space, or a mid-flight gesture/session the engine
  // refuses) means "browser menu suppressed, no layer menu".
  const [layerMenuTarget, setLayerMenuTarget] = useState<CanvasLayerContextMenuTarget | null>(null);
  const closeLayerMenu = useCallback(() => setLayerMenuTarget(null), []);
  const handleSurfaceContextMenu = (event: ReactMouseEvent<HTMLDivElement>) => {
    // Keep the native menu inside inline editors (the text tool's contenteditable
    // overlay), consistent with the surface-focus INLINE_EDIT_SELECTOR.
    if (event.target instanceof Element && event.target.closest(INLINE_EDIT_SELECTOR)) {
      return;
    }
    // Suppress the browser context menu everywhere else on the surface.
    event.preventDefault();
    if (isInteractionLocked) {
      setLayerMenuTarget(null);
      return;
    }
    if (!engine) {
      setLayerMenuTarget(null);
      return;
    }
    // The container and the canvas targets share a top-left origin, so its rect
    // gives the same screen coords the pointer pipeline feeds the viewport.
    const rect = event.currentTarget.getBoundingClientRect();
    const layerId = engine.contextMenuLayerIdAt({ x: event.clientX - rect.left, y: event.clientY - rect.top });
    if (!layerId) {
      setLayerMenuTarget(null);
      return;
    }
    // Select first (one dispatch, matching a left-click select), then open the menu.
    dispatch({ id: layerId, type: 'setCanvasSelectedLayer' });
    setLayerMenuTarget({ layerId, x: event.clientX, y: event.clientY });
  };

  // The bbox tool snaps to a model-dependent grid; the engine is model-agnostic,
  // so read the active generate model's base and feed the grid size in.
  const modelBase = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate') as { model?: { base?: unknown } } | undefined;
    return typeof values?.model?.base === 'string' ? values.model.base : null;
  });
  useEffect(() => {
    engine?.setBboxGrid(gridSizeForModelBase(modelBase));
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
      // Only engine-backed settings feed a store; React-consumed ones (e.g.
      // showProgressOnCanvas, read below) have no store and are skipped here.
      if (setting.store) {
        engine.stores[setting.store].set(settings[setting.key]);
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
    engine?.stores.checkerColors.set(resolveCheckerColors());
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
    hasStagingSlots,
    isCanvasGenerationInFlight,
    operationKind: operation?.status === 'active' ? operation.identity.kind : null,
  });
  const isInteractionLocked = interactionCapabilities.isSurfaceInteractionLocked;

  useLayoutEffect(() => {
    engine?.setInteractionLocked(isInteractionLocked);
    return () => engine?.setInteractionLocked(false);
  }, [engine, isInteractionLocked]);

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
    engine?.setStagedPreview(previewSource);
  });
  useEffect(() => {
    applyStagedPreview();
  }, [engine, previewKey]);
  // Clear the preview when the widget (or engine) goes away, so an accepted /
  // discarded candidate never lingers over the canvas.
  useEffect(() => {
    return () => engine?.setStagedPreview(null);
  }, [engine]);

  const executeCanvasHotkey = useEffectEvent((commandId: string) => {
    const { layers, selectedLayerId } = document;
    const selectedIndex = selectedLayerId ? layers.findIndex((layer) => layer.id === selectedLayerId) : -1;
    const selectedLayer = selectedIndex >= 0 ? layers[selectedIndex] : undefined;

    if ((commandId === 'canvas.prevEntity' || commandId === 'canvas.nudgeLeft') && hasStagingSlots) {
      dispatch({ direction: -1, type: 'cycleStagedImage' });
      return;
    }

    if ((commandId === 'canvas.nextEntity' || commandId === 'canvas.nudgeRight') && hasStagingSlots) {
      dispatch({ direction: 1, type: 'cycleStagedImage' });
      return;
    }

    if (commandId === 'canvas.deleteSelected' && selectedCandidate) {
      dispatch({ type: 'discardSelectedStagedImage' });
      return;
    }

    if (isInteractionLocked) {
      if (commandId === 'canvas.tool.view') {
        engine?.setTool('view');
      }
      return;
    }

    // Arrow-key nudge: engine owns the bounds/lock logic (no-op with no/locked selection).
    const nudge = NUDGE_DELTAS[commandId];
    if (nudge) {
      engine?.nudgeSelectedLayer(nudge.dx, nudge.dy);
      return;
    }

    // Layer z-reorder: same forward/inverse construction as the layers panel.
    const reorderKind = REORDER_KINDS[commandId];
    if (reorderKind) {
      if (!engine || selectedIndex < 0) {
        return;
      }
      const currentIds = layers.map((layer) => layer.id);
      const nextIds = reorderIdsForHotkey(currentIds, selectedIndex, reorderKind);
      if (!nextIds) {
        return;
      }
      const { forward, inverse } = reorderLayerActions(currentIds, nextIds);
      engine.commitStructural(t('widgets.canvas.commands.reorderLayer'), forward, inverse);
      return;
    }

    if (commandId === 'canvas.deleteSelected') {
      // Staged images take priority; otherwise delete the selected layer (undoable).
      if (engine && selectedLayer && selectedIndex >= 0) {
        const { forward, inverse } = deleteLayerActions(selectedLayer, selectedIndex);
        engine.commitStructural(t('widgets.canvas.commands.deleteLayer'), forward, inverse);
      }
    } else if (commandId === 'canvas.resetSelected') {
      if (engine && selectedLayer) {
        engine.clearMask(selectedLayer.id);
      }
    } else if (commandId === 'canvas.undo') {
      // Canvas undo/redo is engine-scoped: it drives the engine-owned pixel/
      // structural history, NOT project-level (reducer) undo. When the canvas
      // history is empty this is a no-op — it deliberately does not fall back to
      // `undoProjectChange` (project undo keeps its own commands/hotkeys, e.g.
      // the workflow editor's `workflows.undo`).
      engine?.undo();
    } else if (commandId === 'canvas.redo') {
      engine?.redo();
    } else if (commandId === 'canvas.tool.view') {
      engine?.setTool('view');
    } else if (commandId === 'canvas.tool.move') {
      engine?.setTool('move');
    } else if (commandId === 'canvas.transformSelected') {
      // Selecting the transform tool opens a session on the selected layer (if any
      // eligible one); Apply/Cancel (enter/esc) are handled engine-side.
      engine?.setTool('transform');
    } else if (commandId === 'canvas.tool.bbox') {
      engine?.setTool('bbox');
    } else if (commandId === 'canvas.tool.brush') {
      engine?.setTool('brush');
    } else if (commandId === 'canvas.tool.eraser') {
      engine?.setTool('eraser');
    } else if (commandId === 'canvas.tool.lasso') {
      engine?.setTool('lasso');
    } else if (commandId === 'canvas.tool.shape') {
      engine?.setTool('shape');
    } else if (commandId === 'canvas.tool.text') {
      engine?.setTool('text');
    } else if (commandId === 'canvas.tool.gradient') {
      engine?.setTool('gradient');
    } else if (commandId === 'canvas.selectAll') {
      engine?.selectAll();
    } else if (commandId === 'canvas.deselect') {
      engine?.deselect();
    } else if (commandId === 'canvas.invertSelection') {
      engine?.invertSelection();
    } else if (commandId === 'canvas.brushSizeDown') {
      engine?.stepBrushSize(-1);
    } else if (commandId === 'canvas.brushSizeUp') {
      engine?.stepBrushSize(1);
    } else if (commandId === 'canvas.duplicateLayer') {
      if (engine && selectedLayer) {
        const { forward, inverse } = duplicateLayerActions(selectedLayer.id, createLayerId());
        engine.commitStructural(t('widgets.canvas.commands.duplicateLayer'), forward, inverse);
      }
    } else if (commandId === 'canvas.mergeDown') {
      // Gate on the SAME predicate the layers panel's context menu uses to
      // enable/disable its "Merge Down" item (`canMergeLayerDown`), so the hotkey
      // can never fire where the menu would refuse — e.g. a mask layer selected,
      // or a mask directly below the selection. `engine.mergeLayerDown` also
      // guards this itself (defense in depth for callers other than this hotkey),
      // but checking here keeps the two surfaces visibly in lockstep.
      if (engine && selectedLayer && canMergeLayerDown(layers, selectedIndex, true)) {
        engine.mergeLayerDown(selectedLayer.id);
      }
    }
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

  return (
    <Box aria-label={t('widgets.canvas.surface')} bg="bg.inset" h="full" overflow="hidden" position="relative" w="full">
      {engine ? (
        <>
          <CanvasSurface engine={engine} onContextMenu={handleSurfaceContextMenu} />
          <ToolStrip engine={engine} isInteractionLocked={isInteractionLocked} />
          <CanvasLayerContextMenu
            dispatch={dispatch}
            engine={engine}
            layers={document.layers}
            target={layerMenuTarget}
            onClose={closeLayerMenu}
          />
        </>
      ) : null}

      {/*
       * Floating bottom-center chrome: the staging bar (when active) stacks
       * directly above the always-present tool options bar — "just like the
       * staging UI". The wrapper is click-through so the canvas stays
       * interactive around the bars; each bar re-enables pointer events.
       */}
      <Stack align="center" bottom="2" gap="2" left="2" pointerEvents="none" position="absolute" right="2" zIndex="3">
        {hasStagingSlots || isCanvasGenerationInFlight ? (
          <StagingBar
            antialiasProgressImages={antialiasProgressImages}
            areThumbnailsVisible={stagingArea.areThumbnailsVisible}
            autoSwitchMode={stagingArea.autoSwitchMode}
            hasMultipleSlots={hasMultipleStagingSlots}
            isGenerating={isCanvasGenerationInFlight}
            isVisible={stagingArea.isVisible}
            selectedCandidate={selectedCandidate}
            selectedImageIndex={stagingArea.selectedImageIndex}
            selectedSlot={selectedSlot}
            slots={stagingSlots}
            onAccept={() => dispatch({ type: 'acceptStagedImage' })}
            onCancelQueueItem={(queueItemId) => dispatch({ queueItemId, type: 'cancelQueueItem' })}
            onCycle={(direction) => dispatch({ direction, type: 'cycleStagedImage' })}
            onDiscardAll={() => dispatch({ type: 'discardAllStagedImages' })}
            onDiscardSelected={() => dispatch({ type: 'discardSelectedStagedImage' })}
            onSelectImage={(imageIndex) => dispatch({ imageIndex, type: 'setStagedImageIndex' })}
            onSetAutoSwitch={(mode) => dispatch({ mode, type: 'setCanvasStagingAutoSwitch' })}
            onToggleThumbnails={() => dispatch({ type: 'toggleCanvasStagingThumbnailsVisibility' })}
            onToggleVisibility={() => dispatch({ type: 'toggleCanvasStagingVisibility' })}
          />
        ) : null}
        {engine && operation?.status === 'active' && interactionCapabilities.isOperationChromeVisible ? (
          <CanvasOperationBar engine={engine} isExternalInteractionLocked={isInteractionLocked} operation={operation} />
        ) : null}
        {engine && interactionCapabilities.isRegularToolOptionsVisible ? (
          <ToolOptionsBar documentHeight={document.height} documentWidth={document.width} engine={engine} />
        ) : null}
      </Stack>
    </Box>
  );
};
