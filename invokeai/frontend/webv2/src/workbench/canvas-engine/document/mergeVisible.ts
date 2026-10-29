import type {
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';
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

/**
 * What generation applies across an inpaint mask's area besides its coverage: the enabled noise level and denoise
 * limit (a disabled modifier keeps its tuned value out of generation). Fill and opacity are display-only.
 */
const maskGenerationSettings = (layer: CanvasInpaintMaskLayerContract): string =>
  `${layer.noise?.isEnabled ? layer.noise.level : '-'}|${layer.denoise?.isEnabled ? layer.denoise.limit : '-'}`;

/**
 * Whether merge-masks can take the exact selection: two or more inpaint masks, each contributing, unlocked and holding
 * pixels, wherever they sit in the stack (`ineligible` otherwise), whose generation settings agree, since the merged
 * mask carries one set over the whole union (`settings-differ`, reported only for an otherwise eligible selection).
 * Regional guidance never merges this way.
 */
export type InpaintMaskMergeEligibility = 'eligible' | 'ineligible' | 'settings-differ';

export const inpaintMaskMergeEligibility = (
  leaves: readonly SemanticLeaf[],
  selectedLayerIds: ReadonlySet<string>,
  hasContent: HasMergeVisibleContent
): InpaintMaskMergeEligibility => {
  if (selectedLayerIds.size < 2) {
    return 'ineligible';
  }
  const settings = new Set<string>();
  let found = 0;
  for (const leaf of leaves) {
    if (!selectedLayerIds.has(leaf.id)) {
      continue;
    }
    if (
      leaf.layer.type !== 'inpaint_mask' ||
      !leaf.contributionEnabled ||
      leaf.effectiveLocked ||
      !hasContent(leaf.id)
    ) {
      return 'ineligible';
    }
    settings.add(maskGenerationSettings(leaf.layer));
    found += 1;
  }
  // A selected group or a missing id is not a mask leaf.
  if (found !== selectedLayerIds.size) {
    return 'ineligible';
  }
  return settings.size === 1 ? 'eligible' : 'settings-differ';
};

export const canMergeInpaintMasks = (
  leaves: readonly SemanticLeaf[],
  selectedLayerIds: ReadonlySet<string>,
  hasContent: HasMergeVisibleContent
): boolean => inpaintMaskMergeEligibility(leaves, selectedLayerIds, hasContent) === 'eligible';
