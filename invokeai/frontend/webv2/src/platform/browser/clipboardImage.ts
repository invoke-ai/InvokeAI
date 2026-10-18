/**
 * The clipboard's image, read without a canvas or widget in sight; inject `clipboard` for tests without real
 * clipboard access.
 */

type ClipboardReadLike = { read(): Promise<readonly ClipboardItem[]> };

export interface ReadClipboardImageDeps {
  clipboard?: ClipboardReadLike;
}

/** MIME types accepted from the clipboard, best first. */
const IMAGE_TYPES = ['image/png', 'image/webp', 'image/jpeg'] as const;

/**
 * The first image on the clipboard, or `null` when it holds none. Returns
 * `null` rather than throwing when the clipboard is unavailable or permission
 * was refused — a paste that finds nothing is an ordinary outcome, not an error.
 */
export const readClipboardImage = async (deps: ReadClipboardImageDeps = {}): Promise<Blob | null> => {
  const clipboard = deps.clipboard ?? (globalThis.navigator?.clipboard as ClipboardReadLike | undefined);
  if (!clipboard?.read) {
    return null;
  }
  let items: readonly ClipboardItem[];
  try {
    items = await clipboard.read();
  } catch {
    // Permission refused, or the document is not focused. Nothing to paste.
    return null;
  }
  for (const item of items) {
    const type = IMAGE_TYPES.find((candidate) => item.types.includes(candidate));
    if (type) {
      return item.getType(type);
    }
  }
  return null;
};
