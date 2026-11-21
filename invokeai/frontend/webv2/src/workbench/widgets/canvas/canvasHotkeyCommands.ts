import type {
  AdjustmentRefusal,
  LayerUnreadable,
  CanvasDocumentContractV3,
  CanvasEngine,
  DocumentRefusal,
  LayerStackMoveKind,
  StructuralCommitResult,
  ToolId,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';

import {
  compileDocumentLeaves,
  getDocumentIndex,
  isGroupNode,
  isNodeHidden,
  isOverlayStack,
} from '@workbench/canvas-engine/api';
import {
  CANVAS_HISTORY_HOTKEYS,
  CANVAS_MERGE_DOWN_HOTKEY,
  executeCanvasToolCommand,
  isCanvasToolCommand,
  mergeSelectedLayerDown,
  type CanvasToolCommandContext,
  type HotkeyTableEntry,
} from '@workbench/widgets/canvas/canvasToolCommands';
import {
  commitPreparedEdit,
  type LayerOperationRefusal,
  type PreparedCommitOutcome,
} from '@workbench/widgets/canvas/useStructuralCommit';
import {
  deleteLayers,
  duplicateLayers,
  groupLayers,
  prepareNewRasterLayer,
  ungroupLayers,
  type LayerCommandDeps,
} from '@workbench/widgets/layers/layerCommands';

/** Keys that act only while the Canvas itself holds focus; rows and fields own them elsewhere. */
export const CANVAS_ONLY_HOTKEYS = [
  ...CANVAS_HISTORY_HOTKEYS,
  // With several results staged or generating `alt+[` / `alt+]` step through them; otherwise they walk the layers.
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
  // Mod+Shift+N is reserved by browsers; creation stays unbound until customized.
  ['canvas.newRasterLayer', 'widgets.canvas.commands.newRasterLayer', [], { ignoreRepeat: true }],
  ['canvas.duplicateLayer', 'widgets.canvas.commands.duplicateLayer', ['mod+j']],
  ['canvas.groupLayers', 'widgets.canvas.commands.groupLayers', ['mod+g']],
  ['canvas.ungroupLayers', 'widgets.canvas.commands.ungroupLayers', ['mod+shift+g']],
  CANVAS_MERGE_DOWN_HOTKEY,
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

/** Tools that keep left/right as nudges while results are staged; elsewhere the arrows step results. */
const STAGING_NUDGE_TOOLS: ReadonlySet<ToolId> = new Set(['move', 'transform']);

/** Command id → z-reorder direction (index 0 = top-most; "forward" moves toward 0). */
const REORDER_KINDS: Record<string, LayerStackMoveKind> = {
  'canvas.layerBackward': 'backward',
  'canvas.layerForward': 'forward',
  'canvas.layerToBack': 'back',
  'canvas.layerToFront': 'front',
};

/** The layer `steps` away in layer order, wrapping; a selected group stands at its first descendant leaf. */
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

/** Q: the bookmark unless already selected, else the previous layer; ids of removed layers are skipped. */
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
  /** Several staging slots: alt+[ / alt+] (and left/right outside Move and Transform) step results. */
  readonly hasMultipleStagingSlots: boolean;
  readonly selectedLayerIds: readonly string[];
  readonly dispatch: CanvasProjectMutationDispatch;
  readonly copySelection: (cut: boolean) => void;
  /** Crops every unlocked layer to the frame and reports the outcome, as the canvas menu does. */
  readonly cropAllLayers: () => void;
  /** Delete, duplicate, group and ungroup run on `selectedLayerIds` through these, as every layer surface does. */
  readonly layerCommands: LayerCommandDeps;
  readonly pasteFromClipboard: () => void;
  readonly reportStructuralCommit: (result: StructuralCommitResult) => void;
  readonly reportPreparedCommit: (outcome: PreparedCommitOutcome) => void;
  readonly reportLayerOperation: (refusal: LayerOperationRefusal | AdjustmentRefusal | LayerUnreadable) => void;
  readonly reportMergeDownRefusal: (refusal: DocumentRefusal) => void;
  readonly t: (key: string) => string;
}

export const executeCanvasHotkeyCommand = (commandId: string, ctx: CanvasHotkeyContext): void => {
  const { dispatch, document, engine, t } = ctx;
  if (commandId === 'canvas.deleteSelected' && engine?.interaction.get('vectorEditSession')) {
    engine.layers.vectorEdit.deleteSelection();
    return;
  }
  const { selectedLayerId } = document;
  const index = getDocumentIndex(document);
  const selected = selectedLayerId ? index.byId.get(selectedLayerId) : undefined;
  const selectedLayer = selected && !isGroupNode(selected.node) ? selected.node : undefined;

  const stepsResults =
    commandId === 'canvas.prevEntity' ||
    commandId === 'canvas.nextEntity' ||
    ((commandId === 'canvas.nudgeLeft' || commandId === 'canvas.nudgeRight') &&
      !STAGING_NUDGE_TOOLS.has(engine?.interaction.get('activeTool') ?? 'move'));
  if (stepsResults && ctx.hasMultipleStagingSlots) {
    const isNext = commandId === 'canvas.nextEntity' || commandId === 'canvas.nudgeRight';
    dispatch({ direction: isNext ? 1 : -1, type: 'cycleStagedImage' });
    return;
  }

  if (isCanvasToolCommand(commandId)) {
    executeCanvasToolCommand(commandId, { ...ctx, selectedLayerId: selectedLayer?.id ?? null });
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
    const vectorSession = engine?.interaction.get('vectorEditSession');
    if (
      engine &&
      vectorSession &&
      (vectorSession.activePathId !== null || vectorSession.selectedPoints.length > 0 || vectorSession.pathTransform)
    ) {
      engine.layers.vectorEdit.nudge(nudge.dx, nudge.dy);
      return;
    }
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
    } else {
      deleteLayers(ctx.layerCommands, ctx.selectedLayerIds);
    }
  } else if (commandId === 'canvas.copySelection' || commandId === 'canvas.cutSelection') {
    ctx.copySelection(commandId === 'canvas.cutSelection');
  } else if (commandId === 'canvas.pasteImage') {
    ctx.pasteFromClipboard();
  } else if (commandId === 'canvas.newRasterLayer') {
    ctx.reportPreparedCommit(
      commitPreparedEdit(engine, t('widgets.layers.actions.addRasterLayer'), prepareNewRasterLayer)
    );
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
    // With a live pixel selection, mod+J is "layer via copy" — it lifts just the selected pixels.
    if (engine?.interaction.get('hasSelection')) {
      const result = engine.selection.liftSelectionToLayer();
      if (result.status !== 'created' && result.status !== 'empty') {
        ctx.reportLayerOperation(result.status);
      }
    } else {
      void duplicateLayers(ctx.layerCommands, ctx.selectedLayerIds);
    }
  } else if (commandId === 'canvas.groupLayers') {
    groupLayers(ctx.layerCommands, ctx.selectedLayerIds);
  } else if (commandId === 'canvas.ungroupLayers') {
    ungroupLayers(ctx.layerCommands, ctx.selectedLayerIds);
  } else if (commandId === 'canvas.mergeDown') {
    mergeSelectedLayerDown(engine, document, ctx.reportLayerOperation, ctx.reportMergeDownRefusal);
  }
};
