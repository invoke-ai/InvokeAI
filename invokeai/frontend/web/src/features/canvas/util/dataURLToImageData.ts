/**
 * Gets an ImageData object from an image dataURL by drawing it to a canvas.
 */
export const dataURLToImageData = async (
  dataURL: string,
  width: number,
  height: number
): Promise<ImageData> =>
  new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = new Image();

    if (!ctx) {
      canvas.remove();
      reject('Unable to get context');
      return;
    }

    image.onload = function () {
      ctx.drawImage(image, 0, 0);
      canvas.remove();
      resolve(ctx.getImageData(0, 0, width, height));
    };

    image.src = dataURL;
  });
