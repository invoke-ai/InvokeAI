import type {
  AdjustmentRefusal,
  LayerUnreadable,
  CanvasDocumentContractV3,
  CanvasEngine,
  LayerStackMoveKind,
  StructuralCommitResult,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';

import {
  compileDocumentLeaves,
  getDocumentIndex,
  isGroupNode,
  isNodeHidden,
  isOverlayStack,
} from '@workbench/canvas-engine/api';
import { publishLayerPanelSelection } from '@workbench/layerPanelState';
import {
  CANVAS_HISTORY_HOTKEYS,
  executeCanvasToolCommand,
  isCanvasToolCommand,
  type CanvasToolCommandContext,
  type HotkeyTableEntry,
} from '@workbench/widgets/canvas/canvasToolCommands';
import {
  commitPreparedEdit,
  type LayerOperationRefusal,
  type PreparedCommitOutcome,
} from '@workbench/widgets/canvas/useStructuralCommit';
import {
  canGroupSelection,
  canRemoveFromGroup,
  canUngroupSelection,
  groupLayers,
  removeFromGroup,
  ungroupLayers,
} from '@workbench/widgets/layers/layerGroupCommands';
import { canMergeLayerDown } from '@workbench/widgets/layers/layerOps';

/** Keys that act only while the Canvas itself holds focus; rows and fields own them elsewhere. */
export const CANVAS_ONLY_HOTKEYS = [
  ...CANVAS_HISTORY_HOTKEYS,
  // Staging keeps `alt+[` / `alt+]`; bare left/right are registered as layer nudges,
  // then intercepted to cycle staging slots while any slot exists.
  ['canvas.prevEntity', 'widgets.canvas.commands.previousEntity', ['alt+[']],
  ['canvas.nextEntity', 'widgets.canvas.commands.nextEntity', ['alt+]']],
  ['canvas.quickSwitch', 'widgets.canvas.commands.quickSwitch', ['q']],
  ['canvas.deleteSelected', 'widgets.canvas.commands.deleteSelected', ['delete', 'backspace']],
  ['canvas.resetSelected', 'widgets.canvas.commands.resetSelected', ['shift+c']],
  ['canvas.toggleNonRasterLayers', 'widgets.canvas.commands.toggleNonRasterLayers', ['shift+h']],
  ['canvas.copySelection', 'widgets.canvas.commands.copySelection', ['mod+c']],
  ['canvas.cutSelection', 'widgets.canvas.commands.cutSelection', ['mod+x']],
  ['canvas.pasteImage', 'widgets.canvas.commands.pasteImage', ['mod+v']],
  // Selection: select all / deselect / invert (engine-owned transient selection).
  ['canvas.selectAll', 'widgets.canvas.commands.selectAll', ['mod+a']],
  ['canvas.deselect', 'widgets.canvas.commands.deselect', ['mod+d']],
  ['canvas.invertSelection', 'widgets.canvas.commands.invertSelection', ['mod+shift+i']],
  // Move the selected layer: arrows nudge 1px, shift+arrows 10px.
  ['canvas.nudgeLeft', 'widgets.canvas.commands.nudgeLeft', ['arrowleft']],
  ['canvas.nudgeRight', 'widgets.canvas.commands.nudgeRight', ['arrowright']],
  ['canvas.nudgeUp', 'widgets.canvas.commands.nudgeUp', ['arrowup']],
  ['canvas.nudgeDown', 'widgets.canvas.commands.nudgeDown', ['arrowdown']],
  ['canvas.nudgeLeftLarge', 'widgets.canvas.commands.nudgeLeftLarge', ['shift+arrowleft']],
  ['canvas.nudgeRightLarge', 'widgets.canvas.commands.nudgeRightLarge', ['shift+arrowright']],
  ['canvas.nudgeUpLarge', 'widgets.canvas.commands.nudgeUpLarge', ['shift+arrowup']],
  ['canvas.nudgeDownLarge', 'widgets.canvas.commands.nudgeDownLarge', ['shift+arrowdown']],
  // Layer management.
  ['canvas.duplicateLayer', 'widgets.canvas.commands.duplicateLayer', ['mod+j']],
  ['canvas.groupLayers', 'widgets.canvas.commands.groupLayers', ['mod+g']],
  ['canvas.ungroupLayers', 'widgets.canvas.commands.ungroupLayers', ['mod+shift+g']],
  ['canvas.mergeDown', 'widgets.canvas.commands.mergeDown', ['mod+e']],
  ['canvas.layerForward', 'widgets.canvas.commands.layerForward', ['mod+]']],
  ['canvas.layerBackward', 'widgets.canvas.commands.layerBackward', ['mod+[']],
  ['canvas.layerToFront', 'widgets.canvas.commands.layerToFront', ['mod+shift+]']],
  ['canvas.layerToBack', 'widgets.canvas.commands.layerToBack', ['mod+shift+[']],
  // No default key — assignable through the hotkeys settings; the canvas menu offers the same action.
  ['canvas.cropAllLayersToFrame', 'widgets.canvas.contextMenu.cropAllLayersToBbox', []],
] as const satisfies readonly HotkeyTableEntry[];

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
const REORDER_KINDS: Record<string, LayerStackMoveKind> = {
  'canvas.layerBackward': 'backward',
  'canvas.layerForward': 'forward',
  'canvas.layerToBack': 'back',
  'canvas.layerToFront': 'front',
};

/**
 * The layer `steps` away from the selection in layer order, wrapping. A selected group stands at its first
 * descendant leaf: next enters the group there, previous leaves it upward. None selected starts from either end.
 */
const cycleLayerId = (document: CanvasDocumentContractV3, steps: 1 | -1): string | null => {
  const { byId, leaves } = getDocumentIndex(document);
  if (leaves.length === 0) {
    return null;
  }
  const { selectedLayerId } = document;
  const selected = selectedLayerId ? byId.get(selectedLayerId) : undefined;
  if (selected && isGroupNode(selected.node)) {
    const first = leaves.findIndex((leaf) => byId.get(leaf.id)?.path.includes(selected.node.id));
    if (first >= 0) {
      return leaves[steps === 1 ? first : (first - 1 + leaves.length) % leaves.length]!.id;
    }
  }
  const current = leaves.findIndex((leaf) => leaf.id === selectedLayerId);
  if (current < 0) {
    return leaves[steps === 1 ? 0 : leaves.length - 1]!.id;
  }
  return leaves[(current + steps + leaves.length) % leaves.length]!.id;
};

/**
 * Q: the bookmarked layer when one is set and not already selected, else the previously selected layer. Ids whose
 * layer no longer exists are skipped rather than cleared, so a removed bookmark never selects nothing.
 */
const quickSwitchLayerId = (document: CanvasDocumentContractV3, engine: CanvasEngine): string | null => {
  const { byId } = getDocumentIndex(document);
  const { bookmarkedLayerId, previousLayerId } = engine.interaction.get('layerQuickSwitch');
  const exists = (id: string | null): id is string => id !== null && byId.has(id);
  if (exists(bookmarkedLayerId) && bookmarkedLayerId !== document.selectedLayerId) {
    return bookmarkedLayerId;
  }
  return exists(previousLayerId) && previousLayerId !== document.selectedLayerId ? previousLayerId : null;
};

/** Pass hotkey dependencies explicitly so dispatch can be tested without mounting React. */
export interface CanvasHotkeyContext extends Omit<CanvasToolCommandContext, 'engine' | 'selectedLayerId'> {
  readonly document: CanvasDocumentContractV3;
  readonly engine: CanvasEngine | null;
  /** Any staging slot exists, so left/right cycle candidates instead of nudging. */
  readonly hasStagingSlots: boolean;
  /** A staged candidate is selected, so Delete discards it instead of touching layers. */
  readonly hasSelectedStagedCandidate: boolean;
  /** Discards the selected staged candidate and offers Undo, as the staging bar's Discard does. */
  readonly discardSelectedStagedImage: () => void;
  readonly selectedLayerIds: readonly string[];
  readonly dispatch: CanvasProjectMutationDispatch;
  readonly copySelection: (cut: boolean) => void;
  /** Crops every unlocked layer to the frame and reports the outcome, as the canvas menu does. */
  readonly cropAllLayers: () => void;
  readonly pasteFromClipboard: () => void;
  readonly notifyLayerDuplicateFailed: () => void;
  readonly reportStructuralCommit: (result: StructuralCommitResult) => void;
  readonly reportPreparedCommit: (outcome: PreparedCommitOutcome) => void;
  readonly reportLayerOperation: (refusal: LayerOperationRefusal | AdjustmentRefusal | LayerUnreadable) => void;
  readonly t: (key: string) => string;
}

/**
 * Routes one canvas command id to its effect.
 *
 * Ordering is load-bearing and reads top-down as a precedence list: staging
 * cycling and staged-candidate discard win over layer commands, companion keys
 * apply their own lock rule, the interaction lock then blocks everything else,
 * and only after those do nudge, reorder, and the per-command table apply.
 */
export const executeCanvasHotkeyCommand = (commandId: string, ctx: CanvasHotkeyContext): void => {
  const { dispatch, document, engine, t } = ctx;
  const { selectedLayerId } = document;
  const index = getDocumentIndex(document);
  const selected = selectedLayerId ? index.byId.get(selectedLayerId) : undefined;
  const selectedLayer = selected && !isGroupNode(selected.node) ? selected.node : undefined;
  const selectedFrozen = !!selected && (selected.ancestorsLocked || selected.node.isLocked);

  if ((commandId === 'canvas.prevEntity' || commandId === 'canvas.nudgeLeft') && ctx.hasStagingSlots) {
    dispatch({ direction: -1, type: 'cycleStagedImage' });
    return;
  }

  if ((commandId === 'canvas.nextEntity' || commandId === 'canvas.nudgeRight') && ctx.hasStagingSlots) {
    dispatch({ direction: 1, type: 'cycleStagedImage' });
    return;
  }

  if (commandId === 'canvas.deleteSelected' && ctx.hasSelectedStagedCandidate) {
    ctx.discardSelectedStagedImage();
    return;
  }

  if (isCanvasToolCommand(commandId)) {
    executeCanvasToolCommand(commandId, { ...ctx, selectedLayerId: selectedLayer?.id ?? null });
    return;
  }

  if (ctx.isInteractionLocked) {
    return;
  }

  // Without staged results, alt+[ / alt+] walk the layers themselves; Q jumps to the bookmark or back.
  if (commandId === 'canvas.prevEntity' || commandId === 'canvas.nextEntity' || commandId === 'canvas.quickSwitch') {
    const target =
      commandId === 'canvas.quickSwitch'
        ? engine && quickSwitchLayerId(document, engine)
        : cycleLayerId(document, commandId === 'canvas.nextEntity' ? 1 : -1);
    if (target !== null && target !== selectedLayerId) {
      dispatch({ id: target, type: 'setCanvasSelectedLayer' });
    }
    return;
  }

  // Arrow-key nudge: engine owns the bounds/lock logic (no-op with no/locked selection).
  const nudge = NUDGE_DELTAS[commandId];
  if (nudge) {
    const nudged = engine?.layers.nudgeSelectedLayer(nudge.dx, nudge.dy);
    // A nudge with nothing eligible selected is an expected no-op; only a broken commit is news.
    if (nudged?.status === 'postcondition-failed') {
      ctx.reportStructuralCommit(nudged);
    }
    return;
  }

  // Layer z-reorder: the same prepared move the layers panel commits.
  const reorderKind = REORDER_KINDS[commandId];
  if (reorderKind) {
    if (!engine || !selected) {
      return;
    }
    ctx.reportPreparedCommit(
      commitPreparedEdit(engine, t('widgets.canvas.commands.reorderLayer'), (model) =>
        model.prepare({ ids: ctx.selectedLayerIds, kind: reorderKind, type: 'move' })
      )
    );
    return;
  }

  if (commandId === 'canvas.deleteSelected') {
    // Delete clears selected pixels before falling back to layer deletion.
    if (engine?.interaction.get('hasSelection')) {
      engine.selection.eraseSelection();
    } else if (engine && selected && !selectedFrozen) {
      ctx.reportPreparedCommit(
        commitPreparedEdit(engine, t('widgets.canvas.commands.deleteLayer'), (model) =>
          model.prepare({ ids: ctx.selectedLayerIds, type: 'remove' })
        )
      );
    }
  } else if (commandId === 'canvas.copySelection' || commandId === 'canvas.cutSelection') {
    ctx.copySelection(commandId === 'canvas.cutSelection');
  } else if (commandId === 'canvas.pasteImage') {
    ctx.pasteFromClipboard();
  } else if (commandId === 'canvas.cropAllLayersToFrame') {
    ctx.cropAllLayers();
  } else if (commandId === 'canvas.toggleNonRasterLayers') {
    // Hide overlays without disabling generation content; showing restores individually hidden nodes too.
    const overlays = index.nodes.filter((entry) => isOverlayStack(entry.stack));
    const leaves = compileDocumentLeaves(document).filter((leaf) => isOverlayStack(leaf.stack));
    if (engine && leaves.length > 0) {
      const nextHidden = leaves.every((leaf) => !leaf.documentHidden);
      const targets = nextHidden
        ? overlays.filter((entry) => entry.parentId === null)
        : overlays.filter((entry) => isNodeHidden(entry.node));
      ctx.reportPreparedCommit(
        commitPreparedEdit(engine, t('widgets.canvas.commands.toggleNonRasterLayers'), (model) =>
          model.prepare({
            type: 'set-hidden',
            updates: targets.map((entry) => ({ id: entry.node.id, isHidden: nextHidden })),
          })
        )
      );
    }
  } else if (commandId === 'canvas.resetSelected') {
    if (engine && selectedLayer) {
      const result = engine.layers.clearMask(selectedLayer.id);
      if (result.status !== 'committed' && result.status !== 'nothing' && result.status !== 'busy') {
        ctx.reportLayerOperation(result.status);
      }
    }
  } else if (commandId === 'canvas.undo') {
    // Use engine pixel/structural history only; empty canvas history never falls back to project undo.
    engine?.history.undo();
  } else if (commandId === 'canvas.redo') {
    engine?.history.redo();
  } else if (commandId === 'canvas.selectAll') {
    engine?.selection.selectAll();
  } else if (commandId === 'canvas.deselect') {
    engine?.selection.deselect();
  } else if (commandId === 'canvas.invertSelection') {
    engine?.selection.invertSelection();
  } else if (commandId === 'canvas.duplicateLayer') {
    // With a live pixel selection, mod+J is "layer via copy" — it lifts just
    // the selected pixels. With none, it duplicates the whole layer.
    if (engine?.interaction.get('hasSelection')) {
      const result = engine.selection.liftSelectionToLayer();
      if (result.status !== 'created' && result.status !== 'empty') {
        ctx.reportLayerOperation(result.status);
      }
    } else if (engine && selected) {
      void engine.layers
        .duplicateLayers(ctx.selectedLayerIds)
        .then((result) => {
          if (result.status === 'duplicated') {
            publishLayerPanelSelection({
              primaryId: result.selectedLayerId,
              projectId: engine.projectId,
              selectedIds: result.duplicateIds,
            });
            return;
          }
          if (result.status === 'busy') {
            return;
          }
          if (result.status === 'nothing') {
            ctx.notifyLayerDuplicateFailed();
            return;
          }
          ctx.reportLayerOperation(result.status === 'unreadable' ? result : result.status);
        })
        .catch(() => {
          // The engine keeps rejected transactions atomic; route both rejection
          // and preflight refusal through the same user-facing command feedback.
          ctx.notifyLayerDuplicateFailed();
        });
    }
  } else if (commandId === 'canvas.groupLayers') {
    if (engine && selected && canGroupSelection(engine.document.model(), ctx.selectedLayerIds)) {
      ctx.reportPreparedCommit(
        groupLayers(engine, engine.projectId, ctx.selectedLayerIds, t('widgets.canvas.commands.groupLayers'))
      );
    }
  } else if (commandId === 'canvas.ungroupLayers') {
    // Groups dissolve; a selection of leaves of one group moves out above it instead.
    const model = engine?.document.model() ?? null;
    if (engine && selected && canUngroupSelection(model, ctx.selectedLayerIds)) {
      ctx.reportPreparedCommit(ungroupLayers(engine, ctx.selectedLayerIds, t('widgets.canvas.commands.ungroupLayers')));
    } else if (engine && selected && canRemoveFromGroup(model, ctx.selectedLayerIds)) {
      ctx.reportPreparedCommit(
        removeFromGroup(engine, ctx.selectedLayerIds, t('widgets.layers.actions.removeFromGroup'))
      );
    }
  } else if (commandId === 'canvas.mergeDown') {
    // Share canMergeLayerDown with the layer menu so shortcuts respect identical eligibility.
    if (engine && selectedLayer && canMergeLayerDown(document, selectedLayer.id, true)) {
      void engine.layers.mergeLayerDown(selectedLayer.id).then(
        (result) => {
          if (result !== 'merged') {
            ctx.reportLayerOperation(result);
          }
        },
        () => ctx.reportLayerOperation('failed')
      );
    }
  }
};
