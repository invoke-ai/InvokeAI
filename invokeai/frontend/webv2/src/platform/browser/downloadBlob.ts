/**
 * Hand a blob to the browser as a download.
 *
 * This was written out longhand in eight places — workflows, projects, PSD
 * export, the gallery, image actions, prompt templates and wildcards — each
 * with its own spelling of the same five lines, and each carrying the same two
 * mistakes.
 *
 * The anchor is appended to the document before it is clicked. A detached
 * anchor works in Chromium but has historically not in Firefox, which is the
 * sort of thing nobody notices until someone reports that export does nothing.
 *
 * The object URL is revoked on the next frame rather than immediately after the
 * click. Revoking synchronously races the browser's own read of the URL, and
 * loses often enough on Safari to produce an empty or failed download; a frame
 * is long enough for the download to have been claimed and short enough that the
 * blob is not left pinned in memory.
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
