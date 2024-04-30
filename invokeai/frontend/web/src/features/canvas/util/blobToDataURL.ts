export const blobToDataURL = (blob: Blob): Promise<string> => {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (_e) => resolve(reader.result as string);
    reader.onerror = (_e) => reject(reader.error);
    reader.onabort = (_e) => reject(new Error('Read aborted'));
    reader.readAsDataURL(blob);
  });
};

export function imageDataToDataURL(imageData: ImageData): string {
  const { width, height } = imageData;

  // Create a canvas to transfer the ImageData to
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  // Draw the ImageData onto the canvas
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Unable to get canvas context');
  }
  ctx.putImageData(imageData, 0, 0);

  // Convert the canvas to a data URL (base64)
  return canvas.toDataURL();
}
