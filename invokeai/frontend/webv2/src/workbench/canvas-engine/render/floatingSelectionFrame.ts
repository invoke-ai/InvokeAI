import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { floatDocumentMatrix } from '@workbench/canvas-engine/selection/floatingSelection';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CompositeOptions } from './compositor';

export interface FloatingSelectionFrame {
  /** What the compositor draws above the layer the pixels were cut from. */
  readonly composite: NonNullable<CompositeOptions['floatingSelection']>;
  /** The layer-local → document matrix the overlay rides its marching ants through. */
  readonly ants: Mat2d;
}

/**
 * Resolve float drawing and ants together so transforms agree. Prefer display-effect pixels for drawing, raw
 * pixels for baking; return null when no float/layer exists.
 */
export const floatingSelectionFrame = (
  float: FloatingSelection | null,
  doc: CanvasDocumentContractV3 | null
): FloatingSelectionFrame | null => {
  const leaf = float && doc ? lookupDocumentLeaf(doc, float.layerId) : null;
  if (!float || !leaf) {
    return null;
  }
  const matrix = bakeMatrix(float.transform);
  const ants = floatDocumentMatrix(leaf.worldTransform, matrix);
  if (!ants) {
    return null;
  }
  return {
    ants,
    composite: {
      layerId: float.layerId,
      matrix,
      rect: float.pixels.rect,
      surface: float.display ?? float.pixels.surface,
    },
  };
};
