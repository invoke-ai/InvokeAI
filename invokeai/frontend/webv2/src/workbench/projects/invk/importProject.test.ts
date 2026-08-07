import { describe, expect, it, vi } from 'vitest';

import type { InvkBoardItem } from './board';

import { binaryEntry, INVK_MAX_ARCHIVE_BYTES, textEntry, writeArchive } from './archive';
import { createArchiveMediaMaterializer, readInvkArchive, restoreArchiveMedia } from './importProject';
import { createRestoredMediaLedger } from './restoreProjectMedia';

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

const boardItem = (overrides: Partial<InvkBoardItem> = {}): InvkBoardItem => ({
  category: 'general',
  kind: 'image',
  name: 'a.png',
  starred: false,
  ...overrides,
});

const boardEntry = (items: InvkBoardItem[] = [boardItem()]) => JSON.stringify({ items, version: 1 });

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

/** The archive this app writes: a document, its bytes, and the project board it came from. */
const boardArchive = (overrides: Record<string, string | Uint8Array> = {}) =>
  archiveFile({
    'board.json': boardEntry(),
    'images/a.png': new Uint8Array([1, 2, 3]),
    'manifest.json': JSON.stringify(manifest()),
    'project.json': JSON.stringify(document()),
    ...overrides,
  });

describe('readInvkArchive', () => {
  it('refuses an oversized file before allocating its bytes', async () => {
    const arrayBuffer = vi.fn(() => Promise.resolve(new ArrayBuffer(0)));
    const file = { arrayBuffer, size: INVK_MAX_ARCHIVE_BYTES + 1 } as unknown as File;

    await expect(readInvkArchive(file)).rejects.toMatchObject({ reason: 'too-large' });
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('returns the manifest, document and bundled images', async () => {
    const contents = await readInvkArchive(await validArchive());

    expect(contents.manifest.name).toBe('My project');
    expect(contents.projectDocument).toEqual(document());
    expect([...contents.images.keys()]).toEqual(['a.png']);
    expect(contents.cover).toBeNull();
  });

  /**
   * "This file names no board" and "this project's board was empty" are different answers, and the
   * restore treats them differently — only the first may have an empty board invented for it. An
   * archive with no `board.json` is one a build from before project boards wrote, not a broken one.
   */
  it('reads an archive with no board enumeration as naming no board', async () => {
    const contents = await readInvkArchive(await validArchive());

    expect(contents.boardSnapshot).toBeNull();
  });

  it('reads and canonicalizes a board enumeration', async () => {
    const contents = await readInvkArchive(
      await boardArchive({
        'board.json': boardEntry([boardItem({ name: 'z.png' }), boardItem({ kind: 'video', name: 'a.mp4' })]),
      })
    );

    // Sorted by kind then name, whatever order the writer used.
    expect(contents.boardSnapshot?.items.map((item) => item.name)).toEqual(['z.png', 'a.mp4']);
  });

  it('reads an empty board without inventing anything', async () => {
    const contents = await readInvkArchive(await boardArchive({ 'board.json': boardEntry([]) }));

    expect(contents.boardSnapshot).toEqual({ items: [], version: 1 });
  });

  it.each([
    ['unparseable', 'not json'],
    ['a duplicate descriptor', JSON.stringify({ items: [boardItem(), boardItem()], version: 1 })],
    ['an unsafe name', JSON.stringify({ items: [boardItem({ name: '../escape.png' })], version: 1 })],
    ['an unknown version', JSON.stringify({ items: [], version: 2 })],
  ])('reports a board enumeration that is %s as damaged', async (_case, contents) => {
    await expect(readInvkArchive(await boardArchive({ 'board.json': contents }))).rejects.toMatchObject({
      reason: 'damaged',
    });
  });

  /** A ZIP path is attacker-controlled text; a media name from the server never has a separator. */
  it('ignores bundled entries whose names are not plain basenames', async () => {
    const contents = await readInvkArchive(
      await validArchive({
        'images/': new Uint8Array([]),
        'images/nested/deep.png': new Uint8Array([4]),
      })
    );

    expect([...contents.images.keys()]).toEqual(['a.png']);
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

  it.each([
    ['an archive with no manifest', () => archiveFile({ 'project.json': '{}' }), 'not-a-project'],
    ['a file that is not a ZIP', () => new File(['not a zip'], 'project.invk'), 'not-a-project'],
    ['a manifest with no document', () => archiveFile({ 'manifest.json': JSON.stringify(manifest()) }), 'damaged'],
    ['an unparseable document', () => validArchive({ 'project.json': 'not json' }), 'damaged'],
    ['a document that is an array', () => validArchive({ 'project.json': '[]' }), 'damaged'],
  ])('refuses %s as %s', async (_label, build, reason) => {
    await expect(readInvkArchive(await build())).rejects.toMatchObject({ name: 'InvkFormatError', reason });
  });
});

describe('createArchiveMediaMaterializer', () => {
  const bytes = (value: number) => new Uint8Array([value]);
  const archive = () => ({
    images: new Map([['a.png', bytes(1)]]),
    videos: new Map([['clip.mp4', bytes(2)]]),
  });

  it('uploads each descriptor to the staging board under its archived category', async () => {
    const uploadBoardImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'fresh.png', width: 1 }));
    const uploadBoardVideo = vi.fn(() => Promise.resolve({ videoName: 'fresh.mp4' }));
    const materialize = createArchiveMediaMaterializer(archive(), { uploadBoardImage, uploadBoardVideo });
    const settled = vi.fn();

    const result = await materialize(
      [boardItem({ category: 'control' }), boardItem({ category: 'user', kind: 'video', name: 'clip.mp4' })],
      'staging',
      settled
    );

    expect(uploadBoardImage).toHaveBeenCalledWith(bytes(1), 'a.png', {
      boardId: 'staging',
      category: 'control',
      contentType: 'image/png',
    });
    expect(uploadBoardVideo).toHaveBeenCalledWith(bytes(2), 'clip.mp4', {
      boardId: 'staging',
      category: 'user',
      contentType: 'video/mp4',
    });
    expect(result.materialized).toEqual([
      { kind: 'image', name: 'fresh.png', sourceName: 'a.png' },
      { kind: 'video', name: 'fresh.mp4', sourceName: 'clip.mp4' },
    ]);
    expect(settled).toHaveBeenCalledTimes(2);
  });

  /** Never an existence check: board media is copied, never adopted. See `restoreProjectMedia`. */
  it('uploads a descriptor whose name the destination may already hold', async () => {
    const uploadBoardImage = vi.fn((_bytes: Uint8Array, fileName: string) =>
      Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
    );
    const result = await createArchiveMediaMaterializer(archive(), { uploadBoardImage })(
      [boardItem()],
      'staging',
      () => undefined
    );

    expect(result.materialized).toEqual([{ kind: 'image', name: 'server-a.png', sourceName: 'a.png' }]);
  });

  it('reports a descriptor the archive carried no bytes for', async () => {
    const uploadBoardImage = vi.fn(() => Promise.reject(new Error('should not be called')));
    const result = await createArchiveMediaMaterializer(archive(), { uploadBoardImage })(
      [boardItem({ name: 'absent.png' })],
      'staging',
      () => undefined
    );

    expect(result.failed).toEqual([{ kind: 'image', name: 'absent.png', reason: 'missing-entry' }]);
    expect(uploadBoardImage).not.toHaveBeenCalled();
  });

  it('reports a rejected upload without abandoning the rest of the board', async () => {
    const uploadBoardImage = vi.fn((_bytes: Uint8Array, fileName: string) =>
      fileName === 'a.png'
        ? Promise.reject(new Error('rejected'))
        : Promise.resolve({ height: 1, imageName: 'fresh.png', width: 1 })
    );
    const withSecond = archive();

    withSecond.images.set('b.png', bytes(3));

    const result = await createArchiveMediaMaterializer(withSecond, { uploadBoardImage })(
      [boardItem(), boardItem({ name: 'b.png' })],
      'staging',
      () => undefined
    );

    expect(result.failed).toEqual([{ kind: 'image', name: 'a.png', reason: 'upload-failed' }]);
    expect(result.materialized).toEqual([{ kind: 'image', name: 'fresh.png', sourceName: 'b.png' }]);
  });
});

describe('restoreArchiveMedia', () => {
  const restoreDeps = (overrides = {}) => ({
    findExistingImageNames: () => Promise.resolve(new Set<string>()),
    findExistingVideoNames: () => Promise.resolve(new Set<string>()),
    starImages: () => Promise.resolve({ failed: [] }),
    starVideos: () => Promise.resolve({ failed: [] }),
    uploadBoardImage: (_bytes: Uint8Array, fileName: string) =>
      Promise.resolve({ height: 1, imageName: `board-${fileName}`, width: 1 }),
    uploadBoardVideo: (_bytes: Uint8Array, fileName: string) => Promise.resolve({ videoName: `board-${fileName}` }),
    uploadImage: (_bytes: Uint8Array, fileName: string) =>
      Promise.resolve({ height: 1, imageName: `loose-${fileName}`, width: 1 }),
    uploadVideo: (_bytes: Uint8Array, fileName: string) => Promise.resolve({ videoName: `loose-${fileName}` }),
    ...overrides,
  });

  /**
   * The overlap case: `a.png` is both on the board and drawn by the canvas. It travels once, as
   * board media, and the layer follows it to the copy.
   */
  it('restores an overlapping item once and points the document at the copy', async () => {
    const archive = await readInvkArchive(await boardArchive());
    const ledger = createRestoredMediaLedger('staging');
    const uploadImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'never', width: 1 }));

    const result = await restoreArchiveMedia(
      archive,
      { boardId: 'staging', ledger, projectDocument: archive.projectDocument, projectId: 'project-new' },
      restoreDeps({ uploadImage })
    );

    expect(uploadImage).not.toHaveBeenCalled();
    expect([...result.mappings.images]).toEqual([['a.png', 'board-a.png']]);
    expect(ledger).toMatchObject({ boardImageNames: ['board-a.png'], imageNames: [] });
  });

  /** Unreferenced board media is exactly what `board.json` exists to carry. */
  it('restores board media the document never mentions', async () => {
    const archive = await readInvkArchive(
      await boardArchive({
        'board.json': boardEntry([boardItem({ name: 'unreferenced.png', starred: true })]),
        'images/unreferenced.png': new Uint8Array([5]),
      })
    );
    const ledger = createRestoredMediaLedger('staging');
    const starImages = vi.fn(() => Promise.resolve({ failed: [] }));

    const result = await restoreArchiveMedia(
      archive,
      { boardId: 'staging', ledger, projectDocument: archive.projectDocument, projectId: 'project-new' },
      restoreDeps({ starImages })
    );

    expect(ledger.boardImageNames).toEqual(['board-unreferenced.png']);
    expect(starImages).toHaveBeenCalledWith(['board-unreferenced.png'], undefined);
    // The document's own `a.png` is not board media here, so it deduplicates as a plain reference.
    expect(result.mappings.images.get('a.png')).toBe('loose-a.png');
    expect(ledger.imageNames).toEqual(['loose-a.png']);
  });

  it('restores an archive with no board as document references only', async () => {
    const archive = await readInvkArchive(await validArchive());
    const ledger = createRestoredMediaLedger(null);

    const result = await restoreArchiveMedia(
      archive,
      { boardId: null, ledger, projectDocument: archive.projectDocument, projectId: 'project-new' },
      restoreDeps()
    );

    expect(result.boardItemIssues).toEqual([]);
    expect(ledger).toMatchObject({ boardImageNames: [], boardVideoNames: [], imageNames: ['loose-a.png'] });
  });

  it('deduplicates a document reference the destination already has', async () => {
    const archive = await readInvkArchive(await validArchive());
    const ledger = createRestoredMediaLedger(null);
    const uploadImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'never', width: 1 }));

    const result = await restoreArchiveMedia(
      archive,
      { boardId: null, ledger, projectDocument: archive.projectDocument, projectId: 'project-new' },
      restoreDeps({ findExistingImageNames: () => Promise.resolve(new Set(['a.png'])), uploadImage })
    );

    expect(uploadImage).not.toHaveBeenCalled();
    expect(result.mappings.images.size).toBe(0);
  });

  it('announces a bundled video as a video, whatever its extension', async () => {
    const archive = await readInvkArchive(
      await boardArchive({
        'board.json': boardEntry([boardItem({ kind: 'video', name: 'clip.unknown' })]),
        'videos/clip.unknown': new Uint8Array([7]),
      })
    );
    const uploadBoardVideo = vi.fn(() => Promise.resolve({ videoName: 'board-clip.mp4' }));

    await restoreArchiveMedia(
      archive,
      {
        boardId: 'staging',
        ledger: createRestoredMediaLedger('staging'),
        projectDocument: archive.projectDocument,
        projectId: 'project-new',
      },
      restoreDeps({ uploadBoardVideo })
    );

    // `image/png` here would be announced to the video endpoint as an image and refused before the
    // bytes were read.
    expect(uploadBoardVideo).toHaveBeenCalledWith(new Uint8Array([7]), 'clip.unknown', {
      boardId: 'staging',
      category: 'general',
      contentType: 'video/mp4',
    });
  });
});
