import { beforeEach, describe, expect, it, vi } from 'vitest';

const transportMocks = vi.hoisted(() => ({ apiFetchRaw: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  apiFetchRaw: transportMocks.apiFetchRaw,
}));

import { INVK_MAX_ARCHIVE_BYTES, readArchive, readEntryText } from './archive';
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
  boardItems: [],
  createdAt: '2026-08-04T00:00:00.000Z',
  name: 'My project',
  projectDocument: projectDocument(),
};

/** The names a plan will fetch, in the union's stable order. */
const plannedNames = (plan: ReturnType<typeof planInvkExport>): string[] => plan.transferItems.map((item) => item.name);

describe('planInvkExport', () => {
  it('bundles the live document and nothing from history', () => {
    expect(plannedNames(planInvkExport(planInput))).toEqual(['live-a.png', 'live-b.png']);
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

  /**
   * The selection was already excluded from bundling; this is the other half —
   * it must not reach the archive at all, or the imported project opens
   * pointing at images the receiving server has never had.
   */
  it('writes a document carrying no gallery selection', () => {
    const plan = planInvkExport({
      ...planInput,
      projectDocument: {
        ...projectDocument(),
        widgetInstances: {
          'gallery-1': {
            state: { values: { selectedImageName: 'image:selected.png', selectedImageNames: ['image:selected.png'] } },
            typeId: 'gallery',
          },
          'preview-1': { state: { values: { compareImage: imageRef('compare.png') } }, typeId: 'preview' },
        },
      },
    });

    expect(plan.documentJson).not.toContain('selected.png');
    expect(plan.documentJson).not.toContain('compare.png');
    expect(plannedNames(plan)).toEqual(['live-a.png', 'live-b.png']);
  });

  it('omits the source project id when the document has none', () => {
    const plan = planInvkExport({ ...planInput, projectDocument: { canvas: {}, layout: {}, name: 'x' } });

    expect(plan.manifestInput.sourceProjectId).toBeUndefined();
    expect(plannedNames(plan)).toEqual([]);
  });
});

describe('executeInvkExport', () => {
  const bytesFor = (imageName: string) => new Uint8Array([...imageName].map((character) => character.codePointAt(0)!));

  beforeEach(() => {
    transportMocks.apiFetchRaw.mockReset();
  });

  it('writes a readable archive holding the manifest, document, images and cover', async () => {
    const download = vi.fn();
    const result = await executeInvkExport(planInvkExport(planInput), {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () =>
        Promise.resolve({ bytes: new Uint8Array([1, 2, 3]), contentType: 'image/webp;charset=binary' }),
    });

    expect(result).toEqual({
      boardItemIssues: [],
      bundledImageCount: 2,
      bundledVideoCount: 0,
      documentReferenceIssues: [],
    });
    expect(download).toHaveBeenCalledTimes(1);

    const [blob, fileName] = download.mock.calls[0]! as [Blob, string];

    expect(fileName).toBe('My project.invk');

    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect([...entries.keys()].sort()).toEqual([
      'board.json',
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
      version: 3,
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

    expect(result).toEqual({
      boardItemIssues: [],
      bundledImageCount: 1,
      bundledVideoCount: 0,
      documentReferenceIssues: [{ kind: 'image', name: 'live-b.png', reason: 'fetch-failed' }],
    });
    expect(download).toHaveBeenCalledTimes(1);
  });

  it('treats a thrown image fetch the same as a missing one', async () => {
    const result = await executeInvkExport(planInvkExport(planInput), {
      download: vi.fn(),
      fetchImageBytes: () => Promise.reject(new Error('network')),
      fetchImageThumbnail: () => Promise.reject(new Error('network')),
    });

    // Every one of these is a document reference, so the loss is a hole in the canvas.
    expect(result.documentReferenceIssues).toEqual([
      { kind: 'image', name: 'live-a.png', reason: 'fetch-failed' },
      { kind: 'image', name: 'live-b.png', reason: 'fetch-failed' },
    ]);
    expect(result.boardItemIssues).toEqual([]);
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

    // `board.json` is written even for a project with no board contents: its absence would be
    // indistinguishable from a v2 archive that never knew about boards.
    expect([...entries.keys()].sort()).toEqual(['board.json', 'manifest.json', 'project.json']);
    expect(JSON.parse(readEntryText(entries.get(INVK_MANIFEST_ENTRY)!)).cover).toBeUndefined();
  });

  it('writes videos under their own prefix, beside the images', async () => {
    const download = vi.fn();
    const plan = planInvkExport({
      ...planInput,
      projectDocument: {
        ...projectDocument(),
        projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'clip.mp4' } } }, id: 'n' }] },
      },
    });

    expect(plan.transferItems.filter((item) => item.kind === 'video').map((item) => item.name)).toEqual(['clip.mp4']);

    const result = await executeInvkExport(plan, {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
      fetchVideoBytes: (videoName) => Promise.resolve(bytesFor(videoName)),
    });

    expect(result).toEqual({
      boardItemIssues: [],
      bundledImageCount: 2,
      bundledVideoCount: 1,
      documentReferenceIssues: [],
    });

    const [blob] = download.mock.calls[0]! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect([...entries.keys()].sort()).toEqual([
      'board.json',
      'images/live-a.png',
      'images/live-b.png',
      'manifest.json',
      'project.json',
      'videos/clip.mp4',
    ]);
  });

  it('skips a video the server will not serve, exactly as it skips an image', async () => {
    const plan = planInvkExport({
      ...planInput,
      projectDocument: {
        ...projectDocument(),
        projectGraph: { nodes: [{ data: { inputs: { video: { video_name: 'clip.mp4' } } }, id: 'n' }] },
      },
    });

    const result = await executeInvkExport(plan, {
      download: vi.fn(),
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
      fetchVideoBytes: () => Promise.resolve(null),
    });

    expect(result).toEqual({
      boardItemIssues: [],
      bundledImageCount: 2,
      bundledVideoCount: 0,
      documentReferenceIssues: [{ kind: 'video', name: 'clip.mp4', reason: 'fetch-failed' }],
    });
  });

  /**
   * A cancelled export must not look like a successful one. Every fetch fails
   * when the signal aborts, so treating those as "the server would not serve
   * it" would pack an archive holding nothing and download it.
   */
  it('rejects rather than downloading an archive when the fetches are aborted', async () => {
    const download = vi.fn();
    const abort = () => Promise.reject(new DOMException('The operation was aborted.', 'AbortError'));

    await expect(
      executeInvkExport(planInvkExport(planInput), {
        download,
        fetchImageBytes: abort,
        fetchImageThumbnail: abort,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });

    expect(download).not.toHaveBeenCalled();
  });

  it('rejects when the signal aborts after the last fetch but before packing', async () => {
    const download = vi.fn();
    const controller = new AbortController();

    await expect(
      executeInvkExport(planInvkExport(planInput), {
        download,
        fetchImageBytes: (imageName) => {
          controller.abort();

          return Promise.resolve(bytesFor(imageName));
        },
        fetchImageThumbnail: () => Promise.resolve(null),
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });

    expect(download).not.toHaveBeenCalled();
  });

  it('rejects rather than downloading when cancellation occurs during packing', async () => {
    const download = vi.fn();
    const controller = new AbortController();

    await expect(
      executeInvkExport(planInvkExport(planInput), {
        download,
        fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
        fetchImageThumbnail: () => Promise.resolve(null),
        onProgress: ({ phase }) => {
          if (phase === 'packing') {
            controller.abort();
          }
        },
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });

    expect(download).not.toHaveBeenCalled();
  });

  it('still treats an unservable asset as a skip, not a cancellation', async () => {
    const download = vi.fn();
    const result = await executeInvkExport(planInvkExport(planInput), {
      download,
      fetchImageBytes: (imageName) =>
        imageName === 'live-a.png' ? Promise.reject(new Error('500')) : Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'live-a.png', reason: 'fetch-failed' }]);
    expect(download).toHaveBeenCalledTimes(1);
  });

  it('rejects a cumulative response refusal and cancels an active sibling fetch', async () => {
    let settleFirstRead: (result: ReadableStreamReadResult<Uint8Array>) => void = () => undefined;
    const firstRead = new Promise<ReadableStreamReadResult<Uint8Array>>((resolve) => {
      settleFirstRead = resolve;
    });
    const firstReader = {
      cancel: vi.fn(() => {
        settleFirstRead({ done: true, value: undefined });

        return Promise.resolve();
      }),
      read: vi.fn(() => firstRead),
      releaseLock: vi.fn(),
    };
    const secondReader = {
      cancel: vi.fn(() => Promise.resolve()),
      read: vi
        .fn()
        .mockResolvedValueOnce({ done: false, value: new Uint8Array([1]) })
        .mockResolvedValueOnce({ done: true, value: undefined }),
      releaseLock: vi.fn(),
    };
    const secondBodyCancel = vi.fn(() => Promise.resolve());
    const secondGetReader = vi.fn(() => {
      // With one budget per request, let the first response finish so the
      // broken export resolves instead of hanging in this regression test.
      settleFirstRead({ done: true, value: undefined });

      return secondReader;
    });
    const responses = [
      {
        body: { cancel: vi.fn(() => Promise.resolve()), getReader: vi.fn(() => firstReader) },
        headers: new Headers({ 'content-length': String(INVK_MAX_ARCHIVE_BYTES) }),
        ok: true,
      },
      {
        body: { cancel: secondBodyCancel, getReader: secondGetReader },
        headers: new Headers({ 'content-length': '1' }),
        ok: true,
      },
    ] as unknown as Response[];

    transportMocks.apiFetchRaw.mockImplementation(() => responses.shift());
    const download = vi.fn();

    await expect(
      executeInvkExport(planInvkExport(planInput), {
        download,
        fetchImageThumbnail: () => Promise.resolve(null),
      })
    ).rejects.toMatchObject({ reason: 'too-large' });

    expect(download).not.toHaveBeenCalled();
    expect(secondGetReader).not.toHaveBeenCalled();
    expect(secondBodyCancel).toHaveBeenCalledTimes(1);
    expect(firstReader.cancel).toHaveBeenCalledTimes(1);
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

/**
 * Version 3's whole point: a project's board is part of the project. A result generated but never
 * placed on canvas is named nowhere in the document, so without `board.json` it simply vanishes.
 */
describe('board membership', () => {
  const bytesFor = (name: string) => new Uint8Array([...name].map((character) => character.codePointAt(0)!));
  const boardItem = (name: string, overrides: Record<string, unknown> = {}) => ({
    category: 'general' as const,
    kind: 'image' as const,
    name,
    starred: false,
    ...overrides,
  });

  const readBoard = async (blob: Blob) => {
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    return JSON.parse(readEntryText(entries.get('board.json')!));
  };

  it('carries board media the document never references', async () => {
    const download = vi.fn();
    const plan = planInvkExport({
      ...planInput,
      boardItems: [boardItem('unreferenced.png', { category: 'user', starred: true })],
    });

    expect(plan.transferItems.map((item) => item.name)).toEqual(['live-a.png', 'live-b.png', 'unreferenced.png']);

    const result = await executeInvkExport(plan, {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    expect(result.bundledImageCount).toBe(3);

    const [blob] = download.mock.calls[0]! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect(entries.has('images/unreferenced.png')).toBe(true);
    // Category and starring travel: they are what the item was, not merely that it existed.
    expect(await readBoard(blob)).toEqual({
      items: [{ category: 'user', kind: 'image', name: 'unreferenced.png', starred: true }],
      version: 1,
    });
  });

  it('fetches an item that is both board media and a reference exactly once', async () => {
    const fetchImageBytes = vi.fn((imageName: string) => Promise.resolve(bytesFor(imageName)));

    await executeInvkExport(planInvkExport({ ...planInput, boardItems: [boardItem('live-a.png')] }), {
      download: vi.fn(),
      fetchImageBytes,
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    expect(fetchImageBytes.mock.calls.filter(([name]) => name === 'live-a.png')).toHaveLength(1);
  });

  it('reports an overlapping failure against both roles', async () => {
    const result = await executeInvkExport(planInvkExport({ ...planInput, boardItems: [boardItem('live-a.png')] }), {
      download: vi.fn(),
      fetchImageBytes: (imageName) =>
        imageName === 'live-a.png' ? Promise.resolve(null) : Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'live-a.png', reason: 'fetch-failed' }]);
    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'live-a.png', reason: 'fetch-failed' }]);
  });

  /** The descriptor stays so import can report the loss; dropping it would forget it silently. */
  it('keeps the descriptor for an item whose bytes could not be fetched', async () => {
    const download = vi.fn();

    await executeInvkExport(planInvkExport({ ...planInput, boardItems: [boardItem('gone.png')] }), {
      download,
      fetchImageBytes: (imageName) =>
        imageName === 'gone.png' ? Promise.resolve(null) : Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    const [blob] = download.mock.calls[0]! as [Blob];
    const board = await readBoard(blob);

    expect(board.items.map((item: { name: string }) => item.name)).toEqual(['gone.png']);
  });

  it('records an empty board rather than omitting the entry', async () => {
    const download = vi.fn();

    await executeInvkExport(planInvkExport(planInput), {
      download,
      fetchImageBytes: (imageName) => Promise.resolve(bytesFor(imageName)),
      fetchImageThumbnail: () => Promise.resolve(null),
    });

    const [blob] = download.mock.calls[0]! as [Blob];

    expect(await readBoard(blob)).toEqual({ items: [], version: 1 });
  });

  /** Refused before a single round trip: discovering it after hundreds would be the wrong order. */
  it('refuses a project that could never be packed, before fetching anything', () => {
    const boardItems = Array.from({ length: 20_001 }, (_unused, index) => boardItem(`image-${index}.png`));

    expect(() => planInvkExport({ ...planInput, boardItems })).toThrowError(/archive entries/u);
  });
});
