type ClipboardLike = { write(items: ClipboardItem[]): Promise<void> };
type ClipboardItemCtor = new (items: Record<string, Blob>) => ClipboardItem;

export interface CopyBlobToClipboardDeps {
  clipboard?: ClipboardLike;
  ClipboardItemCtor?: ClipboardItemCtor;
}

export const copyBlobToClipboard = async (blob: Blob, deps: CopyBlobToClipboardDeps = {}): Promise<void> => {
  const clipboard = deps.clipboard ?? globalThis.navigator?.clipboard;
  const ClipboardItemImpl = deps.ClipboardItemCtor ?? globalThis.ClipboardItem;
  if (!clipboard || !ClipboardItemImpl) {
    throw new Error('Clipboard image copy is unavailable');
  }
  const type = blob.type || 'image/png';
  await clipboard.write([new ClipboardItemImpl({ [type]: blob })]);
};
