import { describe, expect, it, vi } from 'vitest';

import { copyBlobToClipboard } from './layerExportActions';

describe('copyBlobToClipboard', () => {
  it('writes a ClipboardItem using the blob MIME type', async () => {
    const write = vi.fn(() => Promise.resolve());
    const created: Record<string, Blob>[] = [];
    class TestClipboardItem {
      presentationStyle: PresentationStyle = 'unspecified';
      types: string[] = [];
      constructor(items: Record<string, Blob>) {
        created.push(items);
        this.types = Object.keys(items);
      }
      getType(type: string): Promise<Blob> {
        return Promise.resolve(created.at(-1)?.[type] ?? new Blob());
      }
    }
    const blob = new Blob(['pixels'], { type: 'image/png' });

    await copyBlobToClipboard(blob, { ClipboardItemCtor: TestClipboardItem, clipboard: { write } });

    expect(created).toHaveLength(1);
    expect(created[0]?.['image/png']).toBe(blob);
    expect(write).toHaveBeenCalledWith([expect.any(TestClipboardItem)]);
  });

  it('fails when the clipboard API is unavailable', async () => {
    await expect(copyBlobToClipboard(new Blob(['pixels']))).rejects.toThrow('Clipboard image copy is unavailable');
  });
});
