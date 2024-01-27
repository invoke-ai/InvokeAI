import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { Dimensions } from 'features/canvas/store/canvasTypes';
import { CANVAS_GRID_SIZE_FINE } from 'features/canvas/store/constants';

const getScaledBoundingBoxDimensions = (dimensions: Dimensions, optimalDimension: number) => {
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

export default getScaledBoundingBoxDimensions;
