import { RootState } from 'app/store/store';
import { getCanvasBaseLayer, getCanvasStage } from './konvaInstanceProvider';
import { isCanvasMaskLine } from '../store/canvasTypes';
import { log } from 'app/logging/useLogger';
import createMaskStage from './createMaskStage';
import { konvaNodeToImageData } from './konvaNodeToImageData';
import { konvaNodeToBlob } from './konvaNodeToBlob';

const moduleLog = log.child({ namespace: 'getCanvasDataURLs' });

/**
 * Gets Blob and ImageData objects for the base and mask layers
 */
export const getCanvasData = async (state: RootState) => {
  const canvasBaseLayer = getCanvasBaseLayer();
  const canvasStage = getCanvasStage();

  if (!canvasBaseLayer || !canvasStage) {
    moduleLog.error('Unable to find canvas / stage');
    return;
  }

  const {
    layerState: { objects },
    boundingBoxCoordinates,
    boundingBoxDimensions,
    isMaskEnabled,
    shouldPreserveMaskedArea,
  } = state.canvas;

  const boundingBox = {
    ...boundingBoxCoordinates,
    ...boundingBoxDimensions,
  };

  // Clone the base layer so we don't affect the visible base layer
  const clonedBaseLayer = canvasBaseLayer.clone();

  // Scale it to 100% so we get full resolution
  clonedBaseLayer.scale({ x: 1, y: 1 });

  // absolute position is needed to get the bounding box coords relative to the base layer
  const absPos = clonedBaseLayer.getAbsolutePosition();

  const offsetBoundingBox = {
    x: boundingBox.x + absPos.x,
    y: boundingBox.y + absPos.y,
    width: boundingBox.width,
    height: boundingBox.height,
  };

  // For the base layer, use the offset boundingBox
  const baseBlob = await konvaNodeToBlob(clonedBaseLayer, offsetBoundingBox);
  const baseImageData = await konvaNodeToImageData(
    clonedBaseLayer,
    offsetBoundingBox
  );

  // For the mask layer, use the normal boundingBox
  const maskStage = await createMaskStage(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [], // only include mask lines, and only if mask is enabled
    boundingBox,
    shouldPreserveMaskedArea
  );
  const maskBlob = await konvaNodeToBlob(maskStage, boundingBox);
  const maskImageData = await konvaNodeToImageData(maskStage, boundingBox);

  return {
    baseBlob,
    baseImageData,
    maskBlob,
    maskImageData,
  };
};
