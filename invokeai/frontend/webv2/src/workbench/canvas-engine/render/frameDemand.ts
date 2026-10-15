import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { intersect, isEmpty, transformBounds, union } from '@workbench/canvas-engine/math/rect';

import { prepareComposite, type CompositeOptions, type CompositePreparation } from './compositor';

export interface FrameDemandInput {
  readonly document: CanvasDocumentContractV3;
  readonly isolationLayerId?: string | null;
  readonly liveCacheRects?: ReadonlyMap<string, Rect>;
  readonly transformOverrides?: CompositeOptions['transformOverrides'];
  readonly viewport: Rect;
  /** The frame description to plan against; prepared from the other inputs when absent. */
  readonly preparation?: CompositePreparation;
}

/**
 * Document-space bounds of a leaf's committed pixels: its persisted source bounds joined with any larger live
 * cache, through its effective matrix. Null when the leaf has no rasterizable source.
 */
export const committedLeafBounds = (
  leaf: SemanticLeaf,
  matrix: Mat2d,
  document: CanvasDocumentContractV3,
  liveRect: Rect | undefined
): Rect | null => {
  if (!renderableSourceOf(leaf.layer)) {
    return null;
  }
  const sourceRect = getSourceContentRect(leaf.layer, document);
  const localRect =
    liveRect && !isEmpty(liveRect) ? (isEmpty(sourceRect) ? liveRect : union(sourceRect, liveRect)) : sourceRect;
  return transformBounds(matrix, localRect);
};

/** Calculates the drawable raster caches whose transformed pixels intersect the next frame. */
export const calculateActiveFrameLayerIds = (input: FrameDemandInput): Set<string> => {
  const plan =
    input.preparation ??
    prepareComposite(input.document, {
      isolationLayerId: input.isolationLayerId ?? null,
      transformOverrides: input.transformOverrides,
    });
  const active = new Set<string>();
  plan.leaves.forEach((leaf, index) => {
    const bounds =
      plan.bounds?.[index] ??
      committedLeafBounds(leaf, plan.matrices[index]!, input.document, input.liveCacheRects?.get(leaf.id));
    if (bounds && intersect(bounds, input.viewport)) {
      active.add(leaf.id);
    }
  });
  return active;
};
