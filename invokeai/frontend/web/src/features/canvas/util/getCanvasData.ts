import { logger } from 'app/logging/logger';
import { $canvasBaseLayer, $canvasStage } from 'features/canvas/store/canvasNanostore';
import type { CanvasLayerState, Dimensions } from 'features/canvas/store/canvasTypes';
import { isCanvasMaskLine } from 'features/canvas/store/canvasTypes';
import { konvaNodeToImageData } from 'features/canvas/util/konvaNodeToImageData';
import type { Vector2d } from 'konva/lib/types';

import createMaskStage from './createMaskStage';
import { konvaNodeToBlob } from './konvaNodeToBlob';

/**
 * Gets Blob and ImageData objects for the base and mask layers
 */
export const getCanvasData = async (
  layerState: CanvasLayerState,
  boundingBoxCoordinates: Vector2d,
  boundingBoxDimensions: Dimensions,
  isMaskEnabled: boolean,
  shouldPreserveMaskedArea: boolean
) => {
  const log = logger('canvas');

  const canvasBaseLayer = $canvasBaseLayer.get();
  const canvasStage = $canvasStage.get();

  if (!canvasBaseLayer || !canvasStage) {
    log.error('Unable to find canvas / stage');
    return;
  }

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
  const baseImageData = await konvaNodeToImageData(clonedBaseLayer, offsetBoundingBox);

  // For the mask layer, use the normal boundingBox
  const maskStage = await createMaskStage(
    isMaskEnabled ? layerState.objects.filter(isCanvasMaskLine) : [], // only include mask lines, and only if mask is enabled
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
