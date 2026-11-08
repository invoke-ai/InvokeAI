/**
 * The only accessor of the async Clipboard API. Browsers expose it only in a secure context (HTTPS or localhost), so
 * on a plain-HTTP origin `navigator.clipboard` and `ClipboardItem` are absent: callers ask `canCopy*` to disable their
 * control, and the operations reject with `ClipboardUnavailableError` instead of throwing a `TypeError`.
 */

type ClipboardItemConstructor = new (items: Record<string, Blob>) => ClipboardItem;

interface ClipboardLike {
  read?(): Promise<readonly ClipboardItem[]>;
  write?(items: ClipboardItem[]): Promise<void>;
  writeText?(text: string): Promise<void>;
}

/** Stands in for the browser's clipboard in tests; an omitted part falls back to the browser's. */
export interface ClipboardAccess {
  clipboard?: ClipboardLike;
  ClipboardItemCtor?: ClipboardItemConstructor;
}

export class ClipboardUnavailableError extends Error {
  constructor() {
    super('The clipboard is unavailable: browsers allow it only over HTTPS or on localhost.');
    this.name = 'ClipboardUnavailableError';
  }
}

const getClipboard = (access: ClipboardAccess): ClipboardLike | undefined => {
  if (access.clipboard) {
    return access.clipboard;
  }
  try {
    return typeof navigator === 'undefined' ? undefined : (navigator.clipboard ?? undefined);
  } catch {
    return undefined;
  }
};

const getClipboardItem = (access: ClipboardAccess): ClipboardItemConstructor | undefined =>
  access.ClipboardItemCtor ?? (typeof globalThis.ClipboardItem === 'function' ? globalThis.ClipboardItem : undefined);

export const canCopyText = (access: ClipboardAccess = {}): boolean =>
  typeof getClipboard(access)?.writeText === 'function';

export const copyText = async (text: string, access: ClipboardAccess = {}): Promise<void> => {
  const clipboard = getClipboard(access);
  if (typeof clipboard?.writeText !== 'function') {
    throw new ClipboardUnavailableError();
  }
  await clipboard.writeText(text);
};

export const canCopyImage = (access: ClipboardAccess = {}): boolean =>
  typeof getClipboard(access)?.write === 'function' && getClipboardItem(access) !== undefined;

/** Puts an image on the clipboard under its own type, PNG when the blob names none. */
export const copyImage = async (blob: Blob, access: ClipboardAccess = {}): Promise<void> => {
  const clipboard = getClipboard(access);
  const ClipboardItemImpl = getClipboardItem(access);
  if (typeof clipboard?.write !== 'function' || !ClipboardItemImpl) {
    throw new ClipboardUnavailableError();
  }
  await clipboard.write([new ClipboardItemImpl({ [blob.type || 'image/png']: blob })]);
};

/** What the clipboard holds, for paste; rejects as the browser does when permission is refused. */
export const readClipboardItems = (access: ClipboardAccess = {}): Promise<readonly ClipboardItem[]> => {
  const clipboard = getClipboard(access);
  return typeof clipboard?.read === 'function' ? clipboard.read() : Promise.reject(new ClipboardUnavailableError());
};
