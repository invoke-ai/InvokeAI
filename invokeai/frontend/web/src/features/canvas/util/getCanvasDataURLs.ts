import { RootState } from 'app/store/store';
import { getCanvasBaseLayer, getCanvasStage } from './konvaInstanceProvider';
import { isCanvasMaskLine } from '../store/canvasTypes';
import {
  buildMaskStage,
  getStageDataURL,
  getStageImageData,
} from './generateMask';
import { log } from 'app/logging/useLogger';
import {
  areAnyPixelsBlack,
  getImageDataTransparency,
} from 'common/util/arrayBuffer';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { masks } from 'dateformat';

const moduleLog = log.child({ namespace: 'getCanvasDataURLs' });

export const getCanvasDataURLs = (state: RootState) => {
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
    stageScale,
    isMaskEnabled,
    shouldPreserveMaskedArea,
    boundingBoxScaleMethod: boundingBoxScale,
    scaledBoundingBoxDimensions,
  } = state.canvas;

  const boundingBox = {
    ...boundingBoxCoordinates,
    ...boundingBoxDimensions,
  };

  // generationParameters.fit = false;

  // generationParameters.strength = img2imgStrength;

  // generationParameters.invert_mask = shouldPreserveMaskedArea;

  // generationParameters.bounding_box = boundingBox;

  const tempScale = canvasBaseLayer.scale();

  canvasBaseLayer.scale({
    x: 1 / stageScale,
    y: 1 / stageScale,
  });

  const absPos = canvasBaseLayer.getAbsolutePosition();

  const offsetBoundingBox = {
    x: boundingBox.x + absPos.x,
    y: boundingBox.y + absPos.y,
    width: boundingBox.width,
    height: boundingBox.height,
  };

  const { stage: maskStage, offscreenContainer } = buildMaskStage(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
    offsetBoundingBox
  );

  const maskDataURL = maskStage.toDataURL(offsetBoundingBox);

  const maskImageData = maskStage
    .toCanvas()
    .getContext('2d')
    ?.getImageData(
      offsetBoundingBox.x,
      offsetBoundingBox.y,
      offsetBoundingBox.width,
      offsetBoundingBox.height
    );

  offscreenContainer.remove();

  if (!maskImageData) {
    moduleLog.error('Unable to get mask stage context');
    return;
  }

  const baseDataURL = canvasBaseLayer.toDataURL(offsetBoundingBox);

  const ctx = canvasBaseLayer.getContext();

  const baseImageData = ctx.getImageData(
    offsetBoundingBox.x,
    offsetBoundingBox.y,
    offsetBoundingBox.width,
    offsetBoundingBox.height
  );

  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData);

  // const doesMaskHaveBlackPixels = false;
  const doesMaskHaveBlackPixels = areAnyPixelsBlack(maskImageData);

  if (state.system.enableImageDebugging) {
    openBase64ImageInTab([
      { base64: maskDataURL, caption: 'mask sent as init_mask' },
      { base64: baseDataURL, caption: 'image sent as init_img' },
    ]);
  }

  canvasBaseLayer.scale(tempScale);

  // generationParameters.init_img = imageDataURL;
  // generationParameters.progress_images = false;

  // if (boundingBoxScale !== 'none') {
  //   generationParameters.inpaint_width = scaledBoundingBoxDimensions.width;
  //   generationParameters.inpaint_height = scaledBoundingBoxDimensions.height;
  // }

  // generationParameters.seam_size = seamSize;
  // generationParameters.seam_blur = seamBlur;
  // generationParameters.seam_strength = seamStrength;
  // generationParameters.seam_steps = seamSteps;
  // generationParameters.tile_size = tileSize;
  // generationParameters.infill_method = infillMethod;
  // generationParameters.force_outpaint = false;

  return {
    baseDataURL,
    maskDataURL,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  };
};
