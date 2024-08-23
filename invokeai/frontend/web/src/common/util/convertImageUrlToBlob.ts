import { $authToken } from 'app/store/nanostores/authToken';

/**
 * Converts an image URL to a Blob by creating an <img /> element, drawing it to canvas
 * and then converting the canvas to a Blob.
 *
 * @returns A function that takes a URL and returns a Promise that resolves with a Blob
 */

export const convertImageUrlToBlob = (url: string) =>
  new Promise<Blob | null>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;

      const context = canvas.getContext('2d');
      if (!context) {
        reject(new Error('Failed to get canvas context'));
        return;
      }
      context.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert image to blob'));
        }
      }, 'image/png');
    };

    img.onerror = () => {
      reject(new Error('Image failed to load. The URL may be invalid or the object may not exist.'));
    };

    img.crossOrigin = $authToken.get() ? 'use-credentials' : 'anonymous';
    img.src = url;
  });
