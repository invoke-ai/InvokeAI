import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeEntry } from '@workbench/canvas-engine/document/documentIndex';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { isLayerHidden } from '@workbench/canvas-engine/document/layerEligibility';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';

/**
 * Document facts about one leaf, with its ancestors already applied; nothing about the screen,
 * the panel or the session. Ordering belongs to the sequence a leaf is returned in.
 */
export interface SemanticLeaf {
  readonly id: string;
  readonly stack: LayerStackKind;
  /** Ancestor group ids, root first. */
  readonly parentIds: readonly string[];
  readonly layer: CanvasLayerContract;
  /** The leaf and every ancestor are enabled. */
  readonly contributionEnabled: boolean;
  /** The leaf or an ancestor is display-hidden. */
  readonly documentHidden: boolean;
  /** The leaf's own lock, as the panel shows it. */
  readonly locked: boolean;
  /** The leaf or an ancestor is locked; edits refuse on this. */
  readonly effectiveLocked: boolean;
  readonly worldTransform: Mat2d;
}

export const compileSemanticLeaf = (layer: CanvasLayerContract, entry: CanvasNodeEntry): SemanticLeaf => ({
  contributionEnabled: entry.ancestorsEnabled && layer.isEnabled,
  documentHidden: entry.ancestorsHidden || isLayerHidden(layer),
  effectiveLocked: entry.ancestorsLocked || layer.isLocked,
  id: layer.id,
  layer,
  locked: layer.isLocked,
  parentIds: entry.path,
  stack: entry.stack,
  worldTransform: fromTRS(
    { x: layer.transform.x, y: layer.transform.y },
    layer.transform.rotation,
    layer.transform.scaleX,
    layer.transform.scaleY
  ),
});

/** Whether a cached leaf still describes `layer` in the place `entry` gives it. */
export const isSemanticLeafCurrent = (
  leaf: SemanticLeaf,
  layer: CanvasLayerContract,
  entry: CanvasNodeEntry
): boolean =>
  leaf.layer === layer &&
  leaf.contributionEnabled === (entry.ancestorsEnabled && layer.isEnabled) &&
  leaf.documentHidden === (entry.ancestorsHidden || isLayerHidden(layer)) &&
  leaf.effectiveLocked === (entry.ancestorsLocked || layer.isLocked) &&
  leaf.parentIds.length === entry.path.length &&
  leaf.parentIds.every((id, index) => id === entry.path[index]);
