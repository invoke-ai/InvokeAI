export const getImageDataTransparency = (imageData: ImageData) => {
  let isFullyTransparent = true;
  let isPartiallyTransparent = false;
  const len = imageData.data.length;
  let i = 3;
  for (i; i < len; i += 4) {
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

export const areAnyPixelsBlack = (imageData: ImageData) => {
  const len = imageData.data.length;
  let i = 0;
  for (i; i < len; ) {
    if (
      imageData.data[i++] === 255 &&
      imageData.data[i++] === 255 &&
      imageData.data[i++] === 255 &&
      imageData.data[i++] === 255
    ) {
      return true;
    }
  }
  return false;
};
