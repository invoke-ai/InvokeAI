import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { CANVAS_GRID_SIZE_FINE } from 'features/canvas/store/canvasSlice';
import type { Dimensions } from 'features/canvas/store/canvasTypes';

const getScaledBoundingBoxDimensions = (dimensions: Dimensions) => {
  const { width, height } = dimensions;

  const scaledDimensions = { width, height };
  const targetArea = 512 * 512;
  const aspectRatio = width / height;
  let currentArea = width * height;
  let maxDimension = 448;
  while (currentArea < targetArea) {
    maxDimension += CANVAS_GRID_SIZE_FINE;
    if (width === height) {
      scaledDimensions.width = 512;
      scaledDimensions.height = 512;
      break;
    } else {
      if (aspectRatio > 1) {
        scaledDimensions.width = maxDimension;
        scaledDimensions.height = roundToMultiple(
          maxDimension / aspectRatio,
          CANVAS_GRID_SIZE_FINE
        );
      } else if (aspectRatio < 1) {
        scaledDimensions.height = maxDimension;
        scaledDimensions.width = roundToMultiple(
          maxDimension * aspectRatio,
          CANVAS_GRID_SIZE_FINE
        );
      }
      currentArea = scaledDimensions.width * scaledDimensions.height;
    }
  }

  return scaledDimensions;
};

export default getScaledBoundingBoxDimensions;
