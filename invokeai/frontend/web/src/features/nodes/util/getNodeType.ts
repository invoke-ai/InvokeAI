export const getNodeType = (
  baseIsPartiallyTransparent: boolean,
  baseIsFullyTransparent: boolean,
  doesMaskHaveBlackPixels: boolean
): 'txt2img' | `img2img` | 'inpaint' | 'outpaint' => {
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
