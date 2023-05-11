/**
 * Gets a Blob from a canvas.
 */
export const canvasToBlob = async (canvas: HTMLCanvasElement): Promise<Blob> =>
  new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject('Unable to create Blob');
    });
  });
