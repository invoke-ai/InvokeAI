export const getImageDataTransparency = (imageData: ImageData) => {
  let isFullyTransparent = true;
  let isPartiallyTransparent = false;
  const len = imageData.data.length;
  for (let i = 3; i < len; i += 4) {
    if (imageData.data[i] !== 0) {
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
