export function resizeBase64Image(base64Str: string, targetWidth: number, targetHeight: number) {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64Str;

    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        const resizedBase64 = canvas.toDataURL();
        resolve(resizedBase64);
      } else {
        resolve(base64Str);
      }
    };
  });
}
