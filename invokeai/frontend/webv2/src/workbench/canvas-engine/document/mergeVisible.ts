import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';

import { getDocumentIndex } from './documentIndex';
import { isLayerEditable } from './layerEligibility';

export type HasMergeVisibleContent = (layerId: string) => boolean;

const isMergeVisibleRasterLeaf = (leaf: SemanticLeaf, hasContent: HasMergeVisibleContent): boolean =>
  leaf.stack === 'raster' && leaf.contributionEnabled && hasContent(leaf.id);

/** Contributing raster leaves with content, top first. */
export const getMergeVisibleRasterLeaves = (
  leaves: readonly SemanticLeaf[],
  hasContent: HasMergeVisibleContent
): SemanticLeaf[] => leaves.filter((leaf) => isMergeVisibleRasterLeaf(leaf, hasContent));

/** Contributing raster layers with content, top first. */
export const getMergeVisibleRasterLayers = (
  leaves: readonly SemanticLeaf[],
  hasContent: HasMergeVisibleContent
): CanvasLayerContract[] => getMergeVisibleRasterLeaves(leaves, hasContent).map((leaf) => leaf.layer);

/** Whether the raster stack's merge-visible action has at least two contributors; stops at the second. */
export const canMergeVisibleRasters = (
  leaves: readonly SemanticLeaf[],
  hasContent: HasMergeVisibleContent
): boolean => {
  let found = 0;
  for (const leaf of leaves) {
    if (isMergeVisibleRasterLeaf(leaf, hasContent) && ++found === 2) {
      return true;
    }
  }
  return false;
};

/**
 * Destructive merge-selected may only collapse one uninterrupted run of raster siblings: leaves
 * under one parent with nothing between them, since any node in between takes part in the
 * composite between the selected layers.
 */
export const areSelectedRasterLayersContiguous = (
  document: Pick<CanvasDocumentContractV3, 'stacks'>,
  selectedLayerIds: ReadonlySet<string>
): boolean => {
  if (selectedLayerIds.size < 2) {
    return false;
  }
  const index = getDocumentIndex(document);
  const entries = [...selectedLayerIds].map((id) => index.byId.get(id));
  const first = entries[0];
  if (
    !first ||
    entries.some(
      (entry) => !entry || entry.node.type !== 'raster' || entry.stack !== 'raster' || entry.parentId !== first.parentId
    )
  ) {
    return false;
  }
  const positions = entries.map((entry) => entry!.siblingIndex).sort((a, b) => a - b);
  return positions.at(-1)! - positions[0]! + 1 === positions.length;
};

/** Whether merge-selected can flatten the exact selection without changing the rendered composite. */
export const canMergeSelectedRasters = (
  document: Pick<CanvasDocumentContractV3, 'stacks'>,
  leaves: readonly SemanticLeaf[],
  selectedLayerIds: ReadonlySet<string>,
  hasContent: HasMergeVisibleContent
): boolean =>
  areSelectedRasterLayersContiguous(document, selectedLayerIds) &&
  leaves
    .filter((leaf) => selectedLayerIds.has(leaf.id))
    .every(
      (leaf) =>
        leaf.stack === 'raster' &&
        leaf.contributionEnabled &&
        !leaf.effectiveLocked &&
        isLayerEditable(leaf.layer) &&
        leaf.layer.blendMode === 'normal' &&
        hasContent(leaf.id)
    );
