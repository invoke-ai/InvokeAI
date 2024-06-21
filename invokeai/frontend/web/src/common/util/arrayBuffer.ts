export const getImageDataTransparency = (imageData: ImageData) => {
  let isFullyTransparent = true;
  let isPartiallyTransparent = false;
  const len = imageData.data.length;
  for (let i = 3; i < len; i += 4) {
    if (imageData.data[i] === 255) {
      isFullyTransparent = false;
    } else {
      isPartiallyTransparent = true;
    }
    if (!isFullyTransparent && isPartiallyTransparent) {
      return { isFullyTransparent, isPartiallyTransparent };
    }
  }
  return { isFullyTransparent, isPartiallyTransparent };
};

export const areAnyPixelsBlack = (pixels: Uint8ClampedArray) => {
  const len = pixels.length;
  const i = 0;
  for (let i = 0; i < len; i) {
    if (pixels[i++] === 0 && pixels[i++] === 0 && pixels[i++] === 0 && pixels[i++] === 255) {
      return true;
    }
  }
  return false;
};
