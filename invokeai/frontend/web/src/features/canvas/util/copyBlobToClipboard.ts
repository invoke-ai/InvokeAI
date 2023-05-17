/**
 * Copies a blob to the clipboard by calling navigator.clipboard.write().
 */
export const copyBlobToClipboard = (blob: Blob) => {
  navigator.clipboard.write([
    new ClipboardItem({
      [blob.type]: blob,
    }),
  ]);
};
