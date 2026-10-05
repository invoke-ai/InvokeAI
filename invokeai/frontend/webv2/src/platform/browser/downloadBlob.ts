/**
 * Append the download anchor for Firefox compatibility; defer URL revocation until the next frame to avoid racing
 * browser consumption.
 */
export const downloadBlob = (blob: Blob, fileName: string): void => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.download = fileName;
  anchor.rel = 'noopener';
  anchor.style.display = 'none';

  document.body.append(anchor);
  anchor.click();
  anchor.remove();

  requestAnimationFrame(() => {
    URL.revokeObjectURL(url);
  });
};

/** The same, for content already in hand as text. */
export const downloadText = (text: string, fileName: string, type: string): void => {
  downloadBlob(new Blob([text], { type }), fileName);
};
