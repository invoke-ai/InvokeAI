import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { MainModelBase } from 'features/nodes/types/common';
import { getGridSize, getOptimalDimension } from 'features/parameters/util/optimalDimension';

/**
 * Scales the bounding box dimensions to the optimal dimension. The optimal dimensions should be the trained dimension
 * for the model. For example, 1024 for SDXL or 512 for SD1.5.
 * @param dimensions The un-scaled bbox dimensions
 * @param modelBase The base model
 */
export const getScaledBoundingBoxDimensions = (dimensions: Dimensions, modelBase: MainModelBase): Dimensions => {
  const optimalDimension = getOptimalDimension(modelBase);
  const gridSize = getGridSize(modelBase);
  const width = roundToMultiple(dimensions.width, gridSize);
  const height = roundToMultiple(dimensions.height, gridSize);

  const scaledDimensions = { width, height };
  const targetArea = optimalDimension * optimalDimension;
  const aspectRatio = width / height;

  let currentArea = width * height;
  let maxDimension = optimalDimension - gridSize;

  while (currentArea < targetArea) {
    maxDimension += gridSize;
    if (width === height) {
      scaledDimensions.width = optimalDimension;
      scaledDimensions.height = optimalDimension;
      break;
    } else {
      if (aspectRatio > 1) {
        scaledDimensions.width = maxDimension;
        scaledDimensions.height = roundToMultiple(maxDimension / aspectRatio, gridSize);
      } else if (aspectRatio < 1) {
        scaledDimensions.height = maxDimension;
        scaledDimensions.width = roundToMultiple(maxDimension * aspectRatio, gridSize);
      }
      currentArea = scaledDimensions.width * scaledDimensions.height;
    }
  }

  return scaledDimensions;
};

/**
 * Calculate the new width and height that will fit the given aspect ratio, retaining the input area
 * @param ratio The aspect ratio to calculate the new size for
 * @param area The input area
 * @param modelBase The base model
 * @returns The width and height that will fit the given aspect ratio, retaining the input area
 */
export const calculateNewSize = (ratio: number, area: number, modelBase: MainModelBase): Dimensions => {
  const exactWidth = Math.sqrt(area * ratio);
  const exactHeight = exactWidth / ratio;
  const gridSize = getGridSize(modelBase);

  return {
    width: roundToMultiple(exactWidth, gridSize),
    height: roundToMultiple(exactHeight, gridSize),
  };
};
