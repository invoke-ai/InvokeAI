import type { RootState } from 'app/store/store';
import { $canvasBaseLayer } from 'features/canvas/store/canvasNanostore';

import { konvaNodeToBlob } from './konvaNodeToBlob';

/**
 * Get the canvas base layer blob, with or without bounding box according to `shouldCropToBoundingBoxOnSave`
 */
export const getBaseLayerBlob = async (state: RootState, alwaysUseBoundingBox: boolean = false) => {
  const canvasBaseLayer = $canvasBaseLayer.get();

  if (!canvasBaseLayer) {
    throw new Error('Problem getting base layer blob');
  }

  const { shouldCropToBoundingBoxOnSave, boundingBoxCoordinates, boundingBoxDimensions } = state.canvas;

  const clonedBaseLayer = canvasBaseLayer.clone();

  clonedBaseLayer.scale({ x: 1, y: 1 });

  const absPos = clonedBaseLayer.getAbsolutePosition();

  const boundingBox =
    shouldCropToBoundingBoxOnSave || alwaysUseBoundingBox
      ? {
          x: boundingBoxCoordinates.x + absPos.x,
          y: boundingBoxCoordinates.y + absPos.y,
          width: boundingBoxDimensions.width,
          height: boundingBoxDimensions.height,
        }
      : clonedBaseLayer.getClientRect();

  return konvaNodeToBlob(clonedBaseLayer, boundingBox);
};
