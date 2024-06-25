import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { CANVAS_GRID_SIZE_FINE } from 'features/controlLayers/konva/constants';
import type { Size } from 'features/controlLayers/store/types';

/**
 * Scales the bounding box dimensions to the optimal dimension. The optimal dimensions should be the trained dimension
 * for the model. For example, 1024 for SDXL or 512 for SD1.5.
 * @param dimensions The un-scaled bbox dimensions
 * @param optimalDimension The optimal dimension to scale the bbox to
 */
export const getScaledBoundingBoxDimensions = (dimensions: Size, optimalDimension: number): Size => {
  const { width, height } = dimensions;

  const scaledDimensions = { width, height };
  const targetArea = optimalDimension * optimalDimension;
  const aspectRatio = width / height;
  let currentArea = width * height;
  let maxDimension = optimalDimension - CANVAS_GRID_SIZE_FINE;
  while (currentArea < targetArea) {
    maxDimension += CANVAS_GRID_SIZE_FINE;
    if (width === height) {
      scaledDimensions.width = optimalDimension;
      scaledDimensions.height = optimalDimension;
      break;
    } else {
      if (aspectRatio > 1) {
        scaledDimensions.width = maxDimension;
        scaledDimensions.height = roundToMultiple(maxDimension / aspectRatio, CANVAS_GRID_SIZE_FINE);
      } else if (aspectRatio < 1) {
        scaledDimensions.height = maxDimension;
        scaledDimensions.width = roundToMultiple(maxDimension * aspectRatio, CANVAS_GRID_SIZE_FINE);
      }
      currentArea = scaledDimensions.width * scaledDimensions.height;
    }
  }

  return scaledDimensions;
};
