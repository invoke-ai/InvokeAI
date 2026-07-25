import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { floatDocumentMatrix } from '@workbench/canvas-engine/selection/floatingSelection';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CompositeOptions } from './compositor';

export interface FloatingSelectionFrame {
  /** What the compositor draws above the layer the pixels were cut from. */
  readonly composite: NonNullable<CompositeOptions['floatingSelection']>;
  /** The layer-local → document matrix the overlay rides its marching ants through. */
  readonly ants: Mat2d;
}

/**
 * How a floating selection appears this frame, or `null` when there is nothing
 * floating (or its layer has gone).
 *
 * The composite and the overlay must agree exactly — the ants outline the same
 * pixels the compositor drew — so both are resolved from one call rather than
 * each deriving the transform for itself. `display` is preferred over `pixels`
 * for drawing because a layer with a display-only effect shows its adjusted
 * copy; only the bake ever touches the raw pixels.
 */
export const floatingSelectionFrame = (
  float: FloatingSelection | null,
  doc: CanvasDocumentContractV2 | null
): FloatingSelectionFrame | null => {
  const layer = float && doc ? doc.layers.find((candidate) => candidate.id === float.layerId) : undefined;
  if (!float || !layer) {
    return null;
  }
  const matrix = bakeMatrix(float.transform);
  const ants = floatDocumentMatrix(layerMatrix(layer.transform), matrix);
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
