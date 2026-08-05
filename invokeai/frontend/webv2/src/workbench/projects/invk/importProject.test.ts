import { describe, expect, it, vi } from 'vitest';

import { binaryEntry, textEntry, writeArchive } from './archive';
import { InvkFormatError } from './format';
import { readInvkArchive, restoreArchiveImages } from './importProject';

const imageRef = (imageName: string) => ({ height: 64, imageName, width: 64 });

const document = () => ({
  canvas: { document: { layers: [{ id: 'l1', source: { image: imageRef('a.png'), type: 'image' } }] } },
  id: 'project-1',
  layout: {},
  name: 'My project',
});

const manifest = (overrides: Record<string, unknown> = {}) => ({
  appVersion: '7.0',
  contents: 'workbench-project',
  createdAt: '2026-08-04T00:00:00.000Z',
  name: 'My project',
  version: 2,
  ...overrides,
});

const archiveFile = async (entries: Record<string, string | Uint8Array>): Promise<File> => {
  const blob = await writeArchive(
    new Map(
      Object.entries(entries).map(([path, value]) => [
        path,
        typeof value === 'string' ? textEntry(value) : binaryEntry(value),
      ])
    )
  );

  return new File([blob], 'project.invk');
};

const validArchive = (overrides: Record<string, string | Uint8Array> = {}) =>
  archiveFile({
    'images/a.png': new Uint8Array([1, 2, 3]),
    'manifest.json': JSON.stringify(manifest()),
    'project.json': JSON.stringify(document()),
    ...overrides,
  });

describe('readInvkArchive', () => {
  it('returns the manifest, document and bundled images', async () => {
    const contents = await readInvkArchive(await validArchive());

    expect(contents.manifest.name).toBe('My project');
    expect(contents.projectDocument).toEqual(document());
    expect([...contents.images.keys()]).toEqual(['a.png']);
    expect(contents.cover).toBeNull();
  });

  it('resolves the cover entry named by the manifest', async () => {
    const contents = await readInvkArchive(
      await archiveFile({
        'cover.webp': new Uint8Array([9, 9]),
        'manifest.json': JSON.stringify(manifest({ cover: 'cover.webp' })),
        'project.json': JSON.stringify(document()),
      })
    );

    expect(contents.cover).toEqual({ bytes: new Uint8Array([9, 9]), entryName: 'cover.webp' });
  });

  it('refuses a legacy canvas project by name', async () => {
    const file = await archiveFile({
      'canvas_state.json': '{}',
      'manifest.json': JSON.stringify({ appVersion: '6.9', createdAt: '', name: 'Canvas', version: 1 }),
    });

    await expect(readInvkArchive(file)).rejects.toMatchObject({ reason: 'legacy-canvas-project' });
  });

  it('refuses an archive with no manifest', async () => {
    await expect(readInvkArchive(await archiveFile({ 'project.json': '{}' }))).rejects.toMatchObject({
      reason: 'not-a-project',
    });
  });

  it('refuses a file that is not a ZIP', async () => {
    await expect(readInvkArchive(new File(['not a zip'], 'project.invk'))).rejects.toMatchObject({
      reason: 'not-a-project',
    });
  });

  it('reports a manifest without a document as damaged', async () => {
    const file = await archiveFile({ 'manifest.json': JSON.stringify(manifest()) });

    await expect(readInvkArchive(file)).rejects.toMatchObject({ reason: 'damaged' });
  });

  it('reports an unparseable document as damaged', async () => {
    const file = await validArchive({ 'project.json': 'not json' });

    await expect(readInvkArchive(file)).rejects.toBeInstanceOf(InvkFormatError);
    await expect(readInvkArchive(file)).rejects.toMatchObject({ reason: 'damaged' });
  });

  it('reports a document that is an array as damaged', async () => {
    await expect(readInvkArchive(await validArchive({ 'project.json': '[]' }))).rejects.toMatchObject({
      reason: 'damaged',
    });
  });
});

describe('restoreArchiveImages', () => {
  const contents = async (overrides: Partial<Awaited<ReturnType<typeof readInvkArchive>>> = {}) => ({
    ...(await readInvkArchive(await validArchive())),
    ...overrides,
  });

  it('uploads only what the server is missing', async () => {
    const upload = vi.fn((_bytes: Uint8Array, fileName: string) =>
      Promise.resolve({ height: 1, imageName: fileName, width: 1 })
    );

    const result = await restoreArchiveImages(await contents(), {
      findExistingImageNames: () => Promise.resolve(new Set(['a.png'])),
      uploadArchiveImage: upload,
    });

    expect(upload).not.toHaveBeenCalled();
    expect(result).toMatchObject({ danglingImageNames: [], uploadedCount: 0 });
    expect(result.mapping.size).toBe(0);
  });

  it('records the rename when the server assigns a different name', async () => {
    const result = await restoreArchiveImages(await contents(), {
      findExistingImageNames: () => Promise.resolve(new Set()),
      uploadArchiveImage: () => Promise.resolve({ height: 1, imageName: 'server-1.png', width: 1 }),
    });

    expect([...result.mapping]).toEqual([['a.png', 'server-1.png']]);
    expect(result.uploadedCount).toBe(1);
  });

  it('records no rename when the server keeps the name', async () => {
    const result = await restoreArchiveImages(await contents(), {
      findExistingImageNames: () => Promise.resolve(new Set()),
      uploadArchiveImage: (_bytes, fileName) => Promise.resolve({ height: 1, imageName: fileName, width: 1 }),
    });

    expect(result.mapping.size).toBe(0);
    expect(result.uploadedCount).toBe(1);
  });

  it('reports a reference the archive never carried as dangling', async () => {
    const missing = await contents();

    missing.images.delete('a.png');

    const result = await restoreArchiveImages(missing, {
      findExistingImageNames: () => Promise.resolve(new Set()),
      uploadArchiveImage: () => Promise.reject(new Error('should not be called')),
    });

    expect(result.danglingImageNames).toEqual(['a.png']);
    expect(result.uploadedCount).toBe(0);
  });

  it('reports a failed upload as dangling rather than failing the import', async () => {
    const result = await restoreArchiveImages(await contents(), {
      findExistingImageNames: () => Promise.resolve(new Set()),
      uploadArchiveImage: () => Promise.reject(new Error('upload failed')),
    });

    expect(result.danglingImageNames).toEqual(['a.png']);
    expect(result.uploadedCount).toBe(0);
  });

  it('uploads the cover alongside and returns its server name', async () => {
    const withCover = await contents({ cover: { bytes: new Uint8Array([9]), entryName: 'cover.webp' } });
    const upload = vi.fn(() => Promise.resolve({ height: 1, imageName: 'server-cover.png', width: 1 }));

    const result = await restoreArchiveImages(withCover, {
      findExistingImageNames: () => Promise.resolve(new Set(['a.png'])),
      uploadArchiveImage: upload,
    });

    expect(result.coverImageName).toBe('server-cover.png');
    expect(upload).toHaveBeenCalledWith(new Uint8Array([9]), 'cover.webp', {
      contentType: 'image/webp',
      signal: undefined,
    });
  });

  it('survives a cover that fails to upload', async () => {
    const withCover = await contents({ cover: { bytes: new Uint8Array([9]), entryName: 'cover.webp' } });

    const result = await restoreArchiveImages(withCover, {
      findExistingImageNames: () => Promise.resolve(new Set(['a.png'])),
      uploadArchiveImage: () => Promise.reject(new Error('nope')),
    });

    expect(result.coverImageName).toBeNull();
    expect(result.danglingImageNames).toEqual([]);
  });
});
