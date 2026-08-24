import type { CanvasDocumentContractV2, CanvasEngine } from '@workbench/canvas-engine/api';
import type { LayerReorderKind, StructuralActions } from '@workbench/canvasLayerOps';
import type { CanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';

import { isHideableLayer, isLayerHidden } from '@workbench/canvas-engine/api';
import {
  deleteLayerActions,
  duplicateLayerActions,
  reorderLayerActions,
  reorderSelectionWithinGroupsByKind,
} from '@workbench/canvasLayerOps';
import { canMergeLayerDown } from '@workbench/widgets/layers/layerOps';

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

const deleteSelectedLayerActions = (
  layers: CanvasDocumentContractV2['layers'],
  selectedIds: readonly string[],
  selectedLayerId: string
): StructuralActions | null => {
  const selected = new Set(selectedIds);
  const removed = layers.filter((layer) => selected.has(layer.id));
  if (removed.length === 0 || removed.some((layer) => layer.isLocked)) {
    return null;
  }
  if (removed.length === 1) {
    return deleteLayerActions(removed[0]!, layers.indexOf(removed[0]!));
  }
  return {
    forward: { ids: removed.map((layer) => layer.id), type: 'removeCanvasLayers' },
    inverse: {
      add: { index: 0, layers: removed },
      enabledUpdates: [],
      orderedIds: layers.map((layer) => layer.id),
      selectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    },
  };
};

/**
 * Everything the canvas hotkey dispatcher reads or drives. The widget supplies
 * these from its render scope; keeping them as an explicit parameter is what
 * makes the ~35-command dispatch table testable without mounting React.
 */
export interface CanvasHotkeyContext {
  readonly document: CanvasDocumentContractV2;
  readonly engine: CanvasEngine | null;
  /** Any staging slot exists, so left/right cycle candidates instead of nudging. */
  readonly hasStagingSlots: boolean;
  /** A staged candidate is selected, so Delete discards it instead of touching layers. */
  readonly hasSelectedStagedCandidate: boolean;
  readonly isInteractionLocked: boolean;
  readonly selectedLayerIds: readonly string[];
  readonly dispatch: CanvasProjectMutationDispatch;
  readonly copySelection: (cut: boolean) => void;
  readonly pasteFromClipboard: () => void;
  readonly createLayerId: () => string;
  readonly t: (key: string) => string;
}

/**
 * Routes one canvas command id to its effect.
 *
 * Ordering is load-bearing and reads top-down as a precedence list: staging
 * cycling and staged-candidate discard win over layer commands, the interaction
 * lock then blocks everything except the view tool, and only after those do
 * nudge, reorder, and the per-command table apply.
 */
export const executeCanvasHotkeyCommand = (commandId: string, ctx: CanvasHotkeyContext): void => {
  const { createLayerId, dispatch, document, engine, t } = ctx;
  const { layers, selectedLayerId } = document;
  const selectedIndex = selectedLayerId ? layers.findIndex((layer) => layer.id === selectedLayerId) : -1;
  const selectedLayer = selectedIndex >= 0 ? layers[selectedIndex] : undefined;

  if ((commandId === 'canvas.prevEntity' || commandId === 'canvas.nudgeLeft') && ctx.hasStagingSlots) {
    dispatch({ direction: -1, type: 'cycleStagedImage' });
    return;
  }

  if ((commandId === 'canvas.nextEntity' || commandId === 'canvas.nudgeRight') && ctx.hasStagingSlots) {
    dispatch({ direction: 1, type: 'cycleStagedImage' });
    return;
  }

  if (commandId === 'canvas.deleteSelected' && ctx.hasSelectedStagedCandidate) {
    dispatch({ type: 'discardSelectedStagedImage' });
    return;
  }

  if (ctx.isInteractionLocked) {
    if (commandId === 'canvas.tool.view') {
      engine?.tools.setTool('view');
    }
    return;
  }

  // Arrow-key nudge: engine owns the bounds/lock logic (no-op with no/locked selection).
  const nudge = NUDGE_DELTAS[commandId];
  if (nudge) {
    engine?.layers.nudgeSelectedLayer(nudge.dx, nudge.dy);
    return;
  }

  // Layer z-reorder: same forward/inverse construction as the layers panel.
  const reorderKind = REORDER_KINDS[commandId];
  if (reorderKind) {
    if (!engine || selectedIndex < 0) {
      return;
    }
    const currentIds = layers.map((layer) => layer.id);
    const nextIds = reorderSelectionWithinGroupsByKind(layers, ctx.selectedLayerIds, reorderKind);
    if (!nextIds) {
      return;
    }
    const { forward, inverse } = reorderLayerActions(currentIds, nextIds);
    engine.layers.commitStructural(t('widgets.canvas.commands.reorderLayer'), forward, inverse);
    return;
  }

  if (commandId === 'canvas.deleteSelected') {
    // With a live pixel selection, Delete clears the selected PIXELS — the
    // Photoshop meaning. Only with no selection does it delete the layer.
    if (engine?.interaction.get('hasSelection')) {
      engine.selection.eraseSelection();
    } else if (engine && selectedLayer && selectedIndex >= 0 && !selectedLayer.isLocked) {
      const actions = deleteSelectedLayerActions(layers, ctx.selectedLayerIds, selectedLayer.id);
      if (actions) {
        engine.layers.commitStructural(t('widgets.canvas.commands.deleteLayer'), actions.forward, actions.inverse);
      }
    }
  } else if (commandId === 'canvas.copySelection' || commandId === 'canvas.cutSelection') {
    ctx.copySelection(commandId === 'canvas.cutSelection');
  } else if (commandId === 'canvas.pasteImage') {
    ctx.pasteFromClipboard();
  } else if (commandId === 'canvas.toggleNonRasterLayers') {
    // Hide, never disable: this is the "get the overlays out of my way"
    // shortcut, and it must leave the generated image untouched.
    const hideable = layers.filter(isHideableLayer);
    if (engine && hideable.length > 0) {
      const nextHidden = hideable.every((layer) => !isLayerHidden(layer));
      engine.layers.commitStructural(
        t('widgets.canvas.commands.toggleNonRasterLayers'),
        {
          type: 'setCanvasLayersHidden',
          updates: hideable.map((layer) => ({ id: layer.id, isHidden: nextHidden })),
        },
        {
          type: 'setCanvasLayersHidden',
          updates: hideable.map((layer) => ({ id: layer.id, isHidden: isLayerHidden(layer) })),
        }
      );
    }
  } else if (commandId === 'canvas.resetSelected') {
    if (engine && selectedLayer) {
      engine.layers.clearMask(selectedLayer.id);
    }
  } else if (commandId === 'canvas.undo') {
    // Canvas undo/redo is engine-scoped: it drives the engine-owned pixel/
    // structural history, NOT project-level (reducer) undo. When the canvas
    // history is empty this is a no-op — it deliberately does not fall back to
    // `undoProjectChange` (project undo keeps its own commands/hotkeys, e.g.
    // the workflow editor's `workflows.undo`).
    engine?.history.undo();
  } else if (commandId === 'canvas.redo') {
    engine?.history.redo();
  } else if (commandId === 'canvas.tool.view') {
    engine?.tools.setTool('view');
  } else if (commandId === 'canvas.tool.move') {
    engine?.tools.setTool('move');
  } else if (commandId === 'canvas.transformSelected') {
    // Selecting the transform tool opens a session on the selected layer (if any
    // eligible one); Apply/Cancel (enter/esc) are handled engine-side.
    engine?.tools.setTool('transform');
  } else if (commandId === 'canvas.tool.bbox') {
    engine?.tools.setTool('bbox');
  } else if (commandId === 'canvas.tool.brush') {
    engine?.tools.setTool('brush');
  } else if (commandId === 'canvas.tool.eraser') {
    engine?.tools.setTool('eraser');
  } else if (commandId === 'canvas.tool.lasso') {
    if (engine) {
      // Pressing the shortcut while already on the tool cycles its shape
      // (Photoshop-style) rather than re-selecting the tool it is already on.
      if (engine.interaction.get('activeTool') === 'lasso') {
        const lasso = engine.interaction.get('lassoOptions');
        engine.interaction.set('lassoOptions', {
          ...lasso,
          shape: lasso.shape === 'freehand' ? 'polygon' : 'freehand',
        });
      } else {
        engine.tools.setTool('lasso');
      }
    }
  } else if (commandId === 'canvas.tool.marquee') {
    if (engine) {
      if (engine.interaction.get('activeTool') === 'marquee') {
        const marquee = engine.interaction.get('marqueeOptions');
        engine.interaction.set('marqueeOptions', {
          ...marquee,
          kind: marquee.kind === 'rect' ? 'ellipse' : 'rect',
        });
      } else {
        engine.tools.setTool('marquee');
      }
    }
  } else if (commandId === 'canvas.tool.shape') {
    engine?.tools.setTool('shape');
  } else if (commandId === 'canvas.tool.text') {
    engine?.tools.setTool('text');
  } else if (commandId === 'canvas.tool.gradient') {
    engine?.tools.setTool('gradient');
  } else if (commandId === 'canvas.selectAll') {
    engine?.selection.selectAll();
  } else if (commandId === 'canvas.deselect') {
    engine?.selection.deselect();
  } else if (commandId === 'canvas.invertSelection') {
    engine?.selection.invertSelection();
  } else if (commandId === 'canvas.brushSizeDown') {
    engine?.tools.stepBrushSize(-1);
  } else if (commandId === 'canvas.brushSizeUp') {
    engine?.tools.stepBrushSize(1);
  } else if (commandId === 'canvas.duplicateLayer') {
    // With a live pixel selection, mod+J is "layer via copy" — it lifts just
    // the selected pixels. With none, it duplicates the whole layer.
    if (engine?.interaction.get('hasSelection')) {
      engine.selection.liftSelectionToLayer();
    } else if (engine && selectedLayer) {
      const { forward, inverse } = duplicateLayerActions(selectedLayer.id, createLayerId());
      engine.layers.commitStructural(t('widgets.canvas.commands.duplicateLayer'), forward, inverse);
    }
  } else if (commandId === 'canvas.mergeDown') {
    // Gate on the SAME predicate the layers panel's context menu uses to
    // enable/disable its "Merge Down" item (`canMergeLayerDown`), so the hotkey
    // can never fire where the menu would refuse — e.g. a mask layer selected,
    // or a mask directly below the selection. `engine.layers.mergeLayerDown` also
    // guards this itself (defense in depth for callers other than this hotkey),
    // but checking here keeps the two surfaces visibly in lockstep.
    if (engine && selectedLayer && canMergeLayerDown(layers, selectedIndex, true)) {
      engine.layers.mergeLayerDown(selectedLayer.id);
    }
  }
};
