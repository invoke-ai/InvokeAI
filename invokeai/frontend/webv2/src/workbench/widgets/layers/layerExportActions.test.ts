import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it, vi } from 'vitest';

import { copyBlobToClipboard, saveLayerToAssets } from './layerExportActions';

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

describe('saveLayerToAssets', () => {
  it('threads the account signal through export and upload', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const blob = new Blob(['pixels'], { type: 'image/png' });
    const exportLayer = vi.fn(() => Promise.resolve({ blob, status: 'ok' as const }));
    let uploadedFile: File | null = null;
    const upload = vi.fn((file: File) => {
      uploadedFile = file;
      return Promise.resolve();
    });

    await expect(saveLayerToAssets('layer-a', { exportLayer, upload })).resolves.toBe('saved');

    expect(exportLayer).toHaveBeenCalledWith('layer-a', { includeDisabled: true, signal: owner.signal });
    expect(upload).toHaveBeenCalledWith(expect.any(File), 'none', { signal: owner.signal });
    expect(uploadedFile).toMatchObject({ name: 'layer-layer-a.png', type: 'image/png' });
  });

  it('does not upload an account A export that completes after account B activates', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    let resolveExport!: (value: { status: 'empty' }) => void;
    const exportResult = new Promise<{ status: 'empty' }>((resolve) => {
      resolveExport = resolve;
    });
    const exportLayer = vi.fn(() => exportResult);
    const upload = vi.fn(() => Promise.resolve());
    const pendingSave = saveLayerToAssets('layer-a', { exportLayer, upload });

    accountLifecycle.activate('layer-export-b', ':user:layer-export-b');
    resolveExport({ status: 'empty' });

    await expect(pendingSave).resolves.toBe('stale');
    expect(owner.signal.aborted).toBe(true);
    expect(upload).not.toHaveBeenCalled();
  });

  it('quietly drops an account A upload that completes after account B activates', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const exportLayer = vi.fn(() => Promise.resolve({ blob: new Blob(['pixels']), status: 'ok' as const }));
    let resolveUpload!: () => void;
    const uploadResult = new Promise<void>((resolve) => {
      resolveUpload = resolve;
    });
    const upload = vi.fn(() => uploadResult);
    const pendingSave = saveLayerToAssets('layer-a', { exportLayer, upload });
    await vi.waitFor(() => expect(upload).toHaveBeenCalledOnce());

    accountLifecycle.activate('layer-export-b', ':user:layer-export-b');
    resolveUpload();

    await expect(pendingSave).resolves.toBe('stale');
    expect(owner.signal.aborted).toBe(true);
  });
});
