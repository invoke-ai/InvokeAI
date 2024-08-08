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
    async (url: string, dimension?: number) =>
      new Promise<Blob | null>((resolve) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          let width = img.width;
          let height = img.height;

          if (dimension) {
            const aspectRatio = img.width / img.height;
            if (img.width > img.height) {
              width = dimension;
              height = dimension / aspectRatio;
            } else {
              height = dimension;
              width = dimension * aspectRatio;
            }
          }

          canvas.width = width;
          canvas.height = height;

          const context = canvas.getContext('2d');
          if (!context) {
            return;
          }
          context.drawImage(img, 0, 0, width, height);
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
