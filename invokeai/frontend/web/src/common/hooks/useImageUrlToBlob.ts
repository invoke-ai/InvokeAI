import { $authToken } from 'app/store/nanostores/authToken';
import { useCallback } from 'react';

/**
 * Converts an image URL to a Blob by creating an <img /> element, drawing it to canvas
 * and then converting the canvas to a Blob.
 *
 * @returns A function that takes a URL and returns a Promise that resolves with a Blob
 */
export const useImageUrlToBlob = () => {
  const imageUrlToBlob = useCallback(
    async (url: string) =>
      new Promise<Blob | null>((resolve) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;

          const context = canvas.getContext('2d');
          if (!context) {
            return;
          }
          context.drawImage(img, 0, 0);
          resolve(
            new Promise<Blob | null>((resolve) => {
              canvas.toBlob(function (blob) {
                resolve(blob);
              }, 'image/png');
            })
          );
        };
        img.crossOrigin = $authToken.get() ? 'use-credentials' : 'anonymous';
        img.src = url;
      }),
    []
  );

  return imageUrlToBlob;
};
