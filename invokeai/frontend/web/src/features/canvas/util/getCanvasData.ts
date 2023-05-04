import { RootState } from 'app/store/store';
import { getCanvasBaseLayer, getCanvasStage } from './konvaInstanceProvider';
import { isCanvasMaskLine } from '../store/canvasTypes';
import { log } from 'app/logging/useLogger';
import {
  areAnyPixelsBlack,
  getImageDataTransparency,
} from 'common/util/arrayBuffer';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import generateMask from './generateMask';
import { dataURLToImageData } from './dataURLToUint8ClampedArray';

const moduleLog = log.child({ namespace: 'getCanvasDataURLs' });

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

  const baseDataURL = canvasBaseLayer.toDataURL(offsetBoundingBox);

  canvasBaseLayer.scale(tempScale);

  const maskDataURL = generateMask(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
    boundingBox
  );

  const baseImageData = await dataURLToImageData(
    baseDataURL,
    boundingBox.width,
    boundingBox.height
  );

  const maskImageData = await dataURLToImageData(
    maskDataURL,
    boundingBox.width,
    boundingBox.height
  );

  console.log('baseImageData', baseImageData);
  console.log('maskImageData', maskImageData);

  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData.data);

  const doesMaskHaveBlackPixels = areAnyPixelsBlack(maskImageData.data);

  if (state.system.enableImageDebugging) {
    openBase64ImageInTab([
      { base64: maskDataURL, caption: 'mask b64' },
      { base64: baseDataURL, caption: 'image b64' },
    ]);
  }

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
