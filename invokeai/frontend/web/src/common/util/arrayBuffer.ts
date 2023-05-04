export const getImageDataTransparency = (pixels: Uint8ClampedArray) => {
  let isFullyTransparent = true;
  let isPartiallyTransparent = false;
  const len = pixels.length;
  let i = 3;
  for (i; i < len; i += 4) {
    if (pixels[i] === 255) {
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
  let i = 0;
  for (i; i < len; ) {
    if (
      pixels[i++] === 0 &&
      pixels[i++] === 0 &&
      pixels[i++] === 0 &&
      pixels[i++] === 255
    ) {
      return true;
    }
  }
  return false;
};
