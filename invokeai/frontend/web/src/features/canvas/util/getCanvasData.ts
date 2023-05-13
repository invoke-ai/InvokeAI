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
import { dataURLToImageData } from './dataURLToImageData';
import { canvasToBlob } from './canvasToBlob';

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
    isMaskEnabled,
    shouldPreserveMaskedArea,
    boundingBoxScaleMethod: boundingBoxScale,
    scaledBoundingBoxDimensions,
    stageCoordinates,
  } = state.canvas;

  const boundingBox = {
    ...boundingBoxCoordinates,
    ...boundingBoxDimensions,
  };

  // generationParameters.fit = false;

  // generationParameters.strength = img2imgStrength;

  // generationParameters.invert_mask = shouldPreserveMaskedArea;

  // generationParameters.bounding_box = boundingBox;

  // clone the base layer so we don't affect the actual canvas during scaling
  const clonedBaseLayer = canvasBaseLayer.clone();

  // scale to 1 so we get an uninterpolated image
  clonedBaseLayer.scale({ x: 1, y: 1 });

  // absolute position is needed to get the bounding box coords relative to the base layer
  const absPos = clonedBaseLayer.getAbsolutePosition();

  const offsetBoundingBox = {
    x: boundingBox.x + absPos.x,
    y: boundingBox.y + absPos.y,
    width: boundingBox.width,
    height: boundingBox.height,
  };

  // get a dataURL of the bbox'd region (will convert this to an ImageData to check its transparency)
  const baseDataURL = clonedBaseLayer.toDataURL(offsetBoundingBox);

  // get a blob (will upload this as the canvas intermediate)
  const baseBlob = await canvasToBlob(
    clonedBaseLayer.toCanvas(offsetBoundingBox)
  );

  // build a new mask layer and get its dataURL and blob
  const { maskDataURL, maskBlob } = await generateMask(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
    boundingBox
  );

  // convert to ImageData (via pure jank)
  const baseImageData = await dataURLToImageData(
    baseDataURL,
    boundingBox.width,
    boundingBox.height
  );

  // convert to ImageData (via pure jank)
  const maskImageData = await dataURLToImageData(
    maskDataURL,
    boundingBox.width,
    boundingBox.height
  );

  // check transparency
  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData.data);

  // check mask for black
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
    baseBlob,
    maskDataURL,
    maskBlob,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  };
};
