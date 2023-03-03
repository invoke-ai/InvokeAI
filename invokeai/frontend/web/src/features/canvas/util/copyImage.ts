/**
 * Copies an image to the clipboard by drawing it to a canvas and then
 * calling toBlob() on the canvas.
 */
const copyImage = (url: string, width: number, height: number) => {
  const imageElement = document.createElement('img');

  imageElement.addEventListener('load', () => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');

    if (!context) return;

    context.drawImage(imageElement, 0, 0);

    canvas.toBlob((blob) => {
      blob &&
        navigator.clipboard.write([
          new ClipboardItem({
            [blob.type]: blob,
          }),
        ]);
    });

    canvas.remove();
    imageElement.remove();
  });

  imageElement.src = url;
};

export default copyImage;
