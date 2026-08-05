import { describe, expect, it, vi } from 'vitest';

import { readArchive, readEntryText } from './archive';
import { executeInvkExport, planInvkExport } from './exportProject';
import { INVK_DOCUMENT_ENTRY, INVK_MANIFEST_ENTRY } from './format';

const imageRef = (imageName: string) => ({ height: 64, imageName, width: 64 });

const projectDocument = (): Record<string, unknown> => ({
  canvas: {
    document: {
      layers: [
        { id: 'l1', source: { image: imageRef('live-a.png'), type: 'image' } },
        { id: 'l2', source: { bitmap: imageRef('live-b.png'), type: 'paint' } },
      ],
    },
    snapshots: [{ document: { layers: [{ id: 'old', source: { image: imageRef('history.png'), type: 'image' } }] } }],
  },
  id: 'project-1',
  layout: {},
  name: 'My project',
  queue: { items: [{ snapshot: { canvas: { document: { layers: [{ source: { image: imageRef('q.png') } }] } } } }] },
});

const planInput = {
  appVersion: '7.0',
  createdAt: '2026-08-04T00:00:00.000Z',
  name: 'My project',
  projectDocument: projectDocument(),
};

describe('planInvkExport', () => {
  it('bundles the live document and nothing from history', () => {
    expect(planInvkExport(planInput).imageNames).toEqual(['live-a.png', 'live-b.png']);
  });

  it('names the file from the project and records the source project id', () => {
    const plan = planInvkExport(planInput);

    expect(plan.fileName).toBe('My project.invk');
    expect(plan.manifestInput.sourceProjectId).toBe('project-1');
  });

  it('picks the top-most canvas layer as the cover when nothing has been generated', () => {
    expect(planInvkExport(planInput).coverImageName).toBe('live-a.png');
  });

  it('is pure — the same input yields the same plan', () => {
    expect(planInvkExport(planInput)).toEqual(planInvkExport(planInput));
  });

  it('omits the source project id when the document has none', () => {
    const plan = planInvkExport({ ...planInput, projectDocument: { canvas: {}, layout: {}, name: 'x' } });

    expect(plan.manifestInput.sourceProjectId).toBeUndefined();
    expect(plan.imageNames).toEqual([]);
  });
});

describe('executeInvkExport', () => {
  const bytesFor = (imageName: string) => new Uint8Array([...imageName].map((character) => character.codePointAt(0)!));

  it('writes a readable archive holding the manifest, document, images and cover', async () => {
    const download = vi.fn();
    const result = await executeInvkExport(planInvkExport(planInput), {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () =>
        Promise.resolve({ bytes: new Uint8Array([1, 2, 3]), contentType: 'image/webp;charset=binary' }),
    });

    expect(result).toEqual({ bundledCount: 2, missingImageNames: [] });
    expect(download).toHaveBeenCalledTimes(1);

    const [blob, fileName] = download.mock.calls[0]! as [Blob, string];

    expect(fileName).toBe('My project.invk');

    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect([...entries.keys()].sort()).toEqual([
      'cover.webp',
      'images/live-a.png',
      'images/live-b.png',
      'manifest.json',
      'project.json',
    ]);
    expect(JSON.parse(readEntryText(entries.get(INVK_MANIFEST_ENTRY)!))).toMatchObject({
      contents: 'workbench-project',
      cover: 'cover.webp',
      name: 'My project',
      sourceProjectId: 'project-1',
      version: 2,
    });
    expect(JSON.parse(readEntryText(entries.get(INVK_DOCUMENT_ENTRY)!))).toEqual(projectDocument());
  });

  it('skips an image the server will not serve rather than failing the export', async () => {
    const download = vi.fn();
    const result = await executeInvkExport(planInvkExport(planInput), {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(imageName === 'live-b.png' ? null : bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    expect(result).toEqual({ bundledCount: 1, missingImageNames: ['live-b.png'] });
    expect(download).toHaveBeenCalledTimes(1);
  });

  it('treats a thrown image fetch the same as a missing one', async () => {
    const result = await executeInvkExport(planInvkExport(planInput), {
      download: vi.fn(),
      fetchImageBytes: () => Promise.reject(new Error('network')),
      fetchImageThumbnail: () => Promise.reject(new Error('network')),
    });

    expect(result.missingImageNames).toEqual(['live-a.png', 'live-b.png']);
  });

  it('omits the cover entry when there is no cover to fetch', async () => {
    const download = vi.fn();

    await executeInvkExport(planInvkExport({ ...planInput, projectDocument: { id: 'p', layout: {}, name: 'n' } }), {
      download,
      fetchImageBytes: () => Promise.resolve(null),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    const [blob] = download.mock.calls[0]! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect([...entries.keys()].sort()).toEqual(['manifest.json', 'project.json']);
    expect(JSON.parse(readEntryText(entries.get(INVK_MANIFEST_ENTRY)!)).cover).toBeUndefined();
  });

  it('reports progress through bundling and packing', async () => {
    const onProgress = vi.fn();

    await executeInvkExport(planInvkExport(planInput), {
      download: vi.fn(),
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
      onProgress,
    });

    expect(onProgress.mock.calls.map(([progress]) => progress)).toEqual([
      { completed: 1, phase: 'bundling', total: 2 },
      { completed: 2, phase: 'bundling', total: 2 },
      { completed: 2, phase: 'packing', total: 2 },
    ]);
  });
});
