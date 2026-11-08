/**
 * Append the download anchor for Firefox compatibility; defer URL revocation until the next frame to avoid racing
 * browser consumption.
 */
export const downloadBlob = (blob: Blob, fileName: string): Promise<void> => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  try {
    anchor.href = url;
    anchor.download = fileName;
    anchor.rel = 'noopener';
    anchor.style.display = 'none';
    document.body.append(anchor);
    anchor.click();
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  } finally {
    anchor.remove();
  }
  // Keep frame-delayed consumption; bound cleanup when background tabs suspend animation frames.
  return new Promise<void>((resolve) => {
    const done = (): void => {
      clearTimeout(timer);
      cancelAnimationFrame(frame);
      URL.revokeObjectURL(url);
      resolve();
    };
    const timer = setTimeout(done, 1000);
    const frame = requestAnimationFrame(done);
  });
};

/** The same, for content already in hand as text. */
export const downloadText = (text: string, fileName: string, type: string): void => {
  downloadBlob(new Blob([text], { type }), fileName);
};
