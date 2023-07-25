import {
  areAnyPixelsBlack,
  getImageDataTransparency,
} from 'common/util/arrayBuffer';
import { GenerationMode } from '../store/canvasTypes';

export const getCanvasGenerationMode = (
  baseImageData: ImageData,
  maskImageData: ImageData
): GenerationMode => {
  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData.data);

  // check mask for black
  const doesMaskHaveBlackPixels = areAnyPixelsBlack(maskImageData.data);

  if (baseIsPartiallyTransparent) {
    if (baseIsFullyTransparent) {
      return 'txt2img';
    }

    return 'outpaint';
  } else {
    if (doesMaskHaveBlackPixels) {
      return 'inpaint';
    }

    return 'img2img';
  }
};
