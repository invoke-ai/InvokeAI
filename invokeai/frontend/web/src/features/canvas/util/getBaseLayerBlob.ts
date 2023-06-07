import { getCanvasBaseLayer } from './konvaInstanceProvider';
import { RootState } from 'app/store/store';
import { konvaNodeToBlob } from './konvaNodeToBlob';

/**
 * Get the canvas base layer blob, with or without bounding box according to `shouldCropToBoundingBoxOnSave`
 */
export const getBaseLayerBlob = async (state: RootState) => {
  const canvasBaseLayer = getCanvasBaseLayer();

  if (!canvasBaseLayer) {
    return;
  }

  const {
    shouldCropToBoundingBoxOnSave,
    boundingBoxCoordinates,
    boundingBoxDimensions,
  } = state.canvas;

  const clonedBaseLayer = canvasBaseLayer.clone();

  clonedBaseLayer.scale({ x: 1, y: 1 });

  const absPos = clonedBaseLayer.getAbsolutePosition();

  const boundingBox = shouldCropToBoundingBoxOnSave
    ? {
        x: boundingBoxCoordinates.x + absPos.x,
        y: boundingBoxCoordinates.y + absPos.y,
        width: boundingBoxDimensions.width,
        height: boundingBoxDimensions.height,
      }
    : clonedBaseLayer.getClientRect();

  return konvaNodeToBlob(clonedBaseLayer, boundingBox);
};
