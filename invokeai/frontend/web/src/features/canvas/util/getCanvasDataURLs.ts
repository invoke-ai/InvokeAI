import { RootState } from 'app/store/store';
import { getCanvasBaseLayer, getCanvasStage } from './konvaInstanceProvider';
import { isCanvasMaskLine } from '../store/canvasTypes';
import generateMask from './generateMask';
import { log } from 'app/logging/useLogger';
import {
  areAnyPixelsBlack,
  getImageDataTransparency,
  getIsImageDataWhite,
} from 'common/util/arrayBuffer';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';

export const getCanvasDataURLs = (state: RootState) => {
  const canvasBaseLayer = getCanvasBaseLayer();
  const canvasStage = getCanvasStage();

  if (!canvasBaseLayer || !canvasStage) {
    log.error(
      { namespace: 'getCanvasDataURLs' },
      'Unable to find canvas / stage'
    );
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

  const { dataURL: maskDataURL, imageData: maskImageData } = generateMask(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
    {
      x: boundingBox.x + absPos.x,
      y: boundingBox.y + absPos.y,
      width: boundingBox.width,
      height: boundingBox.height,
    }
  );

  const baseDataURL = canvasBaseLayer.toDataURL({
    x: boundingBox.x + absPos.x,
    y: boundingBox.y + absPos.y,
    width: boundingBox.width,
    height: boundingBox.height,
  });

  const ctx = canvasBaseLayer.getContext();

  const baseImageData = ctx.getImageData(
    boundingBox.x + absPos.x,
    boundingBox.y + absPos.y,
    boundingBox.width,
    boundingBox.height
  );

  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData);

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
