import { describe, expect, it, vi } from 'vitest';

import {
  canCopyImage,
  canCopyText,
  ClipboardUnavailableError,
  copyImage,
  copyText,
  readClipboardItems,
} from './clipboard';

class TestClipboardItem {
  constructor(readonly items: Record<string, Blob>) {}
}
const ClipboardItemCtor = TestClipboardItem as unknown as new (items: Record<string, Blob>) => ClipboardItem;

describe('clipboard', () => {
  // Unit tests run without a browser, as a plain-HTTP page does: neither `navigator.clipboard` nor `ClipboardItem`.
  it('reports copying unavailable without the Clipboard API, and rejects with a typed error', async () => {
    expect(canCopyText()).toBe(false);
    expect(canCopyImage()).toBe(false);

    await expect(copyText('x')).rejects.toBeInstanceOf(ClipboardUnavailableError);
    await expect(copyImage(new Blob(['pixels'], { type: 'image/png' }))).rejects.toBeInstanceOf(
      ClipboardUnavailableError
    );
    await expect(readClipboardItems()).rejects.toBeInstanceOf(ClipboardUnavailableError);
  });

  it('copies text when the clipboard can write it', async () => {
    const writeText = vi.fn(() => Promise.resolve());

    expect(canCopyText({ clipboard: { writeText } })).toBe(true);
    await copyText('project-id', { clipboard: { writeText } });

    expect(writeText).toHaveBeenCalledWith('project-id');
  });

  it('needs both a writable clipboard and ClipboardItem to copy an image, and keeps the blob type', async () => {
    const write = vi.fn((_items: ClipboardItem[]) => Promise.resolve());
    const webp = new Blob(['pixels'], { type: 'image/webp' });

    expect(canCopyImage({ clipboard: { write } })).toBe(false);
    expect(canCopyImage({ ClipboardItemCtor, clipboard: { write } })).toBe(true);
    await copyImage(webp, { ClipboardItemCtor, clipboard: { write } });
    await copyImage(new Blob(['pixels']), { ClipboardItemCtor, clipboard: { write } });

    const [[[typed]], [[untyped]]] = write.mock.calls as unknown as [[[TestClipboardItem]], [[TestClipboardItem]]];
    expect(typed.items).toEqual({ 'image/webp': webp });
    expect(Object.keys(untyped.items)).toEqual(['image/png']);
  });

  it('passes on a refused clipboard write as the browser reports it', async () => {
    const refused = new DOMException('Write permission denied.', 'NotAllowedError');

    await expect(copyText('x', { clipboard: { writeText: () => Promise.reject(refused) } })).rejects.toBe(refused);
  });
});
