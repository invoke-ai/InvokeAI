/**
 * Copies a blob to the clipboard by calling navigator.clipboard.write().
 */
export const copyBlobToClipboard = (blob: Promise<Blob> | Blob, type = 'image/png') => {
  navigator.clipboard.write([
    new ClipboardItem({
      [type]: blob,
    }),
  ]);
};
