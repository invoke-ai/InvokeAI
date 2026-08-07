import { createDraftProject } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectBoardItemDTO } from './api';
import type * as coversModule from './covers';
import type * as assetTransportModule from './invk/assetTransport';
import type * as projectFileModule from './projectFile';
import type * as persistenceModule from './syncedPersistence';

/**
 * The `.invk` workflow end to end: export writes an archive that import reads
 * back, an imported project always lands under a fresh id so a file can never
 * overwrite an existing project, and a file that is not ours is refused before
 * anything reaches the server.
 */

const api = vi.hoisted(() => ({
  createProject: vi.fn(),
  deleteClientStateValue: vi.fn(() => Promise.resolve()),
  getClientStateValue: vi.fn(() => Promise.resolve(null)),
  getProject: vi.fn(),
  // Every export enumerates the project's board; most of these cases do not care what is on it.
  getProjectBoardSnapshot: vi.fn((): Promise<{ items: ProjectBoardItemDTO[] }> => Promise.resolve({ items: [] })),
  isProjectNotFoundError: (error: unknown) =>
    typeof error === 'object' && error !== null && 'status' in error && error.status === 404,
  setClientStateValue: vi.fn(() => Promise.resolve()),
}));

const downloads = vi.hoisted(() => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

const covers = vi.hoisted(() => ({ recordProjectCover: vi.fn() }));

const transport = vi.hoisted(() => ({
  coverExtensionForMime: () => 'webp',
  createAssetExportTransport: () => ({
    fetchImageBytes: transport.fetchImageBytes,
    fetchImageThumbnail: transport.fetchImageThumbnail,
    fetchVideoBytes: transport.fetchVideoBytes,
  }),
  createStagingBoard: vi.fn(() => Promise.resolve('staging-board')),
  deleteArchiveImages: vi.fn(() => Promise.resolve()),
  deleteArchiveVideos: vi.fn(() => Promise.resolve()),
  deleteStagingBoard: vi.fn(() => Promise.resolve()),
  // `Uint8Array | null` up front: a fetcher that returns `null` is how the
  // "server would not serve it" path is exercised, and inferring the narrower
  // type here would make that untypeable at the call site.
  fetchImageBytes: vi.fn((imageName: string): Promise<Uint8Array | null> =>
    Promise.resolve(new TextEncoder().encode(`bytes:${imageName}`))
  ),
  fetchImageThumbnail: vi.fn(() => Promise.resolve(null)),
  fetchVideoBytes: vi.fn((videoName: string): Promise<Uint8Array | null> =>
    Promise.resolve(new TextEncoder().encode(`bytes:${videoName}`))
  ),
  findExistingImageNames: vi.fn((_names: readonly string[]) => Promise.resolve(new Set<string>())),
  findExistingVideoNames: vi.fn((_names: readonly string[]) => Promise.resolve(new Set<string>())),
  mimeForEntryName: () => 'image/png',
  starImages: vi.fn((_names: readonly string[]) => Promise.resolve({ failed: [] as string[] })),
  starVideos: vi.fn((_names: readonly string[]) => Promise.resolve({ failed: [] as string[] })),
  uploadArchiveImage: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
  ),
  uploadArchiveVideo: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ videoName: `server-${fileName}` })
  ),
  uploadBoardImage: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `board-${fileName}`, width: 1 })
  ),
  uploadBoardVideo: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ videoName: `board-${fileName}` })
  ),
}));

vi.mock('./api', () => api);
vi.mock('./covers', async (importOriginal) => ({
  ...(await importOriginal<typeof coversModule>()),
  recordProjectCover: covers.recordProjectCover,
}));
vi.mock('@platform/browser/downloadBlob', () => downloads);
// Partial, so the module's pure predicates stay real. `isRequestCancellation` decides whether a
// failure is this asset's or the whole operation's — a stub of it would let the tests agree with a
// restore that mistook a cancelled import for three hundred dangling references.
vi.mock('./invk/assetTransport', async (importOriginal) => ({
  ...(await importOriginal<typeof assetTransportModule>()),
  ...transport,
}));

let projectFile: typeof projectFileModule;
let persistence: typeof persistenceModule;

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const rasterImageLayer = (id: string, imageName: string) => ({
  blendMode: 'normal' as const,
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 1, imageName, width: 1 }, type: 'image' as const },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster' as const,
});

const projectWithRestorableAssets = (includeVideo = true) => {
  const project = createDraftProject([]);

  return {
    ...project,
    canvas: {
      ...project.canvas,
      document: {
        ...project.canvas.document,
        layers: [rasterImageLayer('restored-image', 'archive-image.png')],
      },
    },
    ...(includeVideo ? { futureVideoInput: { video_name: 'archive-video.mp4' } } : {}),
  };
};

const capturedArchive = (): File => {
  const [blob, fileName] = downloads.downloadBlob.mock.calls.at(-1)! as [Blob, string];

  return new File([blob], fileName);
};

/** The server answers with the board it claimed, or with the one it created for a boardless create. */
const acceptCreate = (): void => {
  api.createProject.mockImplementation(
    (request: { board_id?: string; data?: Record<string, unknown>; name: string; project_id?: string }) =>
      Promise.resolve({
        board_id: request.board_id ?? 'server-created-board',
        created_at: '2026-06-10 10:00:00.000',
        data: request.data ?? {},
        name: request.name,
        project_id: request.project_id ?? '',
        revision: 1,
        updated_at: '2026-06-10 10:00:00.000',
      })
  );
};

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  transport.fetchImageBytes.mockImplementation((imageName: string) =>
    Promise.resolve(new TextEncoder().encode(`bytes:${imageName}`))
  );
  transport.fetchImageThumbnail.mockImplementation(() => Promise.resolve(null));
  transport.findExistingImageNames.mockImplementation((_names: readonly string[]) =>
    Promise.resolve(new Set<string>())
  );
  transport.findExistingVideoNames.mockImplementation((_names: readonly string[]) =>
    Promise.resolve(new Set<string>())
  );
  transport.uploadArchiveImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
  );
  transport.uploadArchiveVideo.mockImplementation((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ videoName: `server-${fileName}` })
  );
  transport.uploadBoardImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `board-${fileName}`, width: 1 })
  );
  transport.uploadBoardVideo.mockImplementation((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ videoName: `board-${fileName}` })
  );
  transport.starImages.mockImplementation((_names: readonly string[]) => Promise.resolve({ failed: [] as string[] }));
  transport.starVideos.mockImplementation((_names: readonly string[]) => Promise.resolve({ failed: [] as string[] }));
  transport.createStagingBoard.mockImplementation(() => Promise.resolve('staging-board'));
  transport.deleteStagingBoard.mockImplementation(() => Promise.resolve());
  transport.deleteArchiveImages.mockImplementation(() => Promise.resolve());
  transport.deleteArchiveVideos.mockImplementation(() => Promise.resolve());
  api.getProjectBoardSnapshot.mockImplementation(() => Promise.resolve({ items: [] }));
  api.getClientStateValue.mockImplementation(() => Promise.resolve(null));

  projectFile = await import('./projectFile');
  persistence = await import('./syncedPersistence');
});

describe('exportOpenProject', () => {
  it('downloads an archive named after the project', async () => {
    await projectFile.exportOpenProject({ ...createDraftProject([]), name: 'My project' });

    expect(downloads.downloadBlob).toHaveBeenCalledTimes(1);
    expect(downloads.downloadBlob.mock.calls[0]![1]).toBe('My project.invk');
  });

  /**
   * The transport is mocked by module path, and every assertion below about what
   * did *not* reach the server is vacuous if that path stops resolving. This one
   * fails loudly instead: a document with a known image reference must reach the
   * mock on the way out.
   */
  it('reaches the server through the mocked transport', async () => {
    api.getProject.mockResolvedValue({
      data: {
        canvas: { document: { layers: [{ id: 'l', source: { image: { imageName: 'pinned.png' }, type: 'image' } }] } },
        id: 'p1',
        layout: {},
        name: 'Pinned',
      },
      name: 'Pinned',
      project_id: 'p1',
      revision: 1,
    });

    await projectFile.exportLibraryProject('p1');

    expect(transport.fetchImageBytes).toHaveBeenCalledWith('pinned.png', expect.anything());
  });
});

describe('exportLibraryProject', () => {
  it('exports a closed project straight from its server record', async () => {
    const document = persistence.serializeProjectDocument({ ...createDraftProject([]), name: 'Closed' });

    api.getProject.mockResolvedValue({ data: document, name: 'Closed', project_id: 'p1', revision: 3 });

    await projectFile.exportLibraryProject('p1');

    expect(api.getProject).toHaveBeenCalledWith('p1', expect.anything());
    expect(downloads.downloadBlob.mock.calls[0]![1]).toBe('Closed.invk');
  });

  /**
   * This is reachable from a project card and from the gallery board menu, both of which can be on
   * screen while the editor holds that project — and the board most likely to be right-clicked is
   * the open project's own. Reading the server record without flushing first hands someone a file
   * missing everything since the last autosave.
   */
  it('flushes an open project before reading its record', async () => {
    const { registerOpenProject, unregisterOpenProject } = await import('./syncStore');
    const order: string[] = [];
    const flush = vi.fn(() => {
      order.push('flush');
      return Promise.resolve();
    });

    api.getProject.mockImplementation(() => {
      order.push('get');

      return Promise.resolve({
        data: persistence.serializeProjectDocument({ ...createDraftProject([]), name: 'Open' }),
        name: 'Open',
        project_id: 'p1',
        revision: 3,
      });
    });
    registerOpenProject('p1', {
      close: vi.fn(),
      flush,
      markDeleted: vi.fn(),
      rename: vi.fn(),
      unmarkDeleted: vi.fn(),
    });

    try {
      await projectFile.exportLibraryProject('p1');
    } finally {
      unregisterOpenProject('p1');
    }

    expect(flush).toHaveBeenCalledTimes(1);
    expect(order).toEqual(['flush', 'get']);
  });

  it('does not flush a project nothing holds', async () => {
    api.getProject.mockResolvedValue({
      data: persistence.serializeProjectDocument({ ...createDraftProject([]), name: 'Closed' }),
      name: 'Closed',
      project_id: 'p1',
      revision: 3,
    });

    await expect(projectFile.exportLibraryProject('p1')).resolves.toBeDefined();
  });
});

/**
 * Both directions can half-succeed, and both used to compute exactly what was
 * lost and then discard it — so a project that shed forty layers looked like a
 * clean round trip. These pin the reporting all the way out to the caller.
 */
describe('what a transfer reports', () => {
  const projectWithImages = () => {
    const project = createDraftProject([]);

    return {
      ...project,
      canvas: {
        ...project.canvas,
        document: {
          ...project.canvas.document,
          layers: [rasterImageLayer('l1', 'a.png'), rasterImageLayer('l2', 'b.png')],
        },
      },
      name: 'Two layers',
    };
  };

  it('counts every asset as it is bundled, then reports packing', async () => {
    const document = { ...persistence.serializeProjectDocument(createDraftProject([])), ...projectWithImages() };
    const onProgress = vi.fn();

    api.getProject.mockResolvedValue({ data: document, name: 'Two layers', project_id: 'p1', revision: 1 });

    await projectFile.exportLibraryProject('p1', { onProgress });

    const phases = onProgress.mock.calls.map(([progress]) => progress.phase);

    expect(phases.filter((phase: string) => phase === 'bundling')).toHaveLength(2);
    expect(phases.at(-1)).toBe('packing');
  });

  it('names the assets the server would not serve', async () => {
    const document = { ...persistence.serializeProjectDocument(createDraftProject([])), ...projectWithImages() };

    api.getProject.mockResolvedValue({ data: document, name: 'Two layers', project_id: 'p1', revision: 1 });
    transport.fetchImageBytes.mockImplementation((imageName: string) =>
      Promise.resolve(imageName === 'b.png' ? null : new TextEncoder().encode('bytes'))
    );

    const outcome = await projectFile.exportLibraryProject('p1');

    expect(outcome.documentReferenceIssues).toEqual([{ kind: 'image', name: 'b.png', reason: 'fetch-failed' }]);
    expect(outcome.fileName).toBe('Two layers.invk');
  });

  it('reports nothing lost for a clean export', async () => {
    const document = { ...persistence.serializeProjectDocument(createDraftProject([])), ...projectWithImages() };

    api.getProject.mockResolvedValue({ data: document, name: 'Two layers', project_id: 'p1', revision: 1 });

    const outcome = await projectFile.exportLibraryProject('p1');

    expect(outcome.documentReferenceIssues).toEqual([]);
    expect(outcome.boardItemIssues).toEqual([]);
  });

  it('counts uploads on the way in, and names what stayed dangling', async () => {
    const document = { ...persistence.serializeProjectDocument(createDraftProject([])), ...projectWithImages() };
    const onProgress = vi.fn();

    acceptCreate();
    api.getProject.mockResolvedValue({ data: document, name: 'Two layers', project_id: 'p1', revision: 1 });
    await projectFile.exportLibraryProject('p1');

    const archive = capturedArchive();

    // Neither image is on the receiving server, and one of them will not upload.
    transport.uploadArchiveImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
      fileName === 'b.png'
        ? Promise.reject(new Error('rejected'))
        : Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
    );

    const outcome = await projectFile.importProjectFile(archive, { onProgress });

    expect(outcome.documentReferenceIssues).toEqual([{ kind: 'image', name: 'b.png', reason: 'upload-failed' }]);
    expect(onProgress.mock.calls.map(([progress]) => progress.phase)).toContain('restoring');
  });
});

/**
 * The board half: an archive carries the project's board, and importing it gives that board's media
 * new identities on a staging board the create then claims. Nothing here may reuse a name the
 * destination already holds — `board_images` keys on the image name, so a reused name would move a
 * stranger's picture onto this project's board instead of copying it.
 */
describe('importing a project board', () => {
  const boardProject = () => {
    const project = createDraftProject([]);

    return {
      ...project,
      canvas: {
        ...project.canvas,
        document: { ...project.canvas.document, layers: [rasterImageLayer('l1', 'shared.png')] },
      },
      name: 'Board project',
    };
  };

  const boardSnapshot = (): { items: ProjectBoardItemDTO[] } => ({
    items: [
      { category: 'general', kind: 'image', name: 'shared.png', starred: true },
      { category: 'user', kind: 'image', name: 'unreferenced.png', starred: false },
      { category: 'general', kind: 'video', name: 'clip.mp4', starred: false },
    ],
  });

  const exportedBoardArchive = async (): Promise<File> => {
    api.getProjectBoardSnapshot.mockResolvedValue(boardSnapshot());
    await projectFile.exportOpenProject(boardProject());

    return capturedArchive();
  };

  const uploadedBoardNames = () => transport.uploadBoardImage.mock.calls.map(([, fileName]) => fileName);

  it('stages the board, restores its media under fresh names, and claims it on create', async () => {
    acceptCreate();

    const archive = await exportedBoardArchive();
    const outcome = await projectFile.importProjectFile(archive);

    expect(transport.createStagingBoard).toHaveBeenCalledWith('Board project', expect.anything());
    expect(uploadedBoardNames().sort()).toEqual(['shared.png', 'unreferenced.png']);
    expect(transport.uploadBoardImage).toHaveBeenCalledWith(expect.anything(), 'unreferenced.png', {
      boardId: 'staging-board',
      category: 'user',
      contentType: 'image/png',
      signal: expect.anything(),
    });
    expect(transport.uploadBoardVideo).toHaveBeenCalledWith(expect.anything(), 'clip.mp4', expect.anything());
    expect(api.createProject.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board' });
    expect(outcome.boardItemIssues).toEqual([]);
    expect(outcome.documentReferenceIssues).toEqual([]);
  });

  it('stars only the descriptors that were starred', async () => {
    acceptCreate();
    await projectFile.importProjectFile(await exportedBoardArchive());

    expect(transport.starImages).toHaveBeenCalledWith(['board-shared.png'], expect.anything());
    expect(transport.starVideos).toHaveBeenCalledWith([], expect.anything());
  });

  it('rewrites the document onto the copy rather than the archived name', async () => {
    acceptCreate();
    await projectFile.importProjectFile(await exportedBoardArchive());

    const { data } = api.createProject.mock.calls[0]![0] as {
      data: { canvas: { document: { layers: Array<{ source: { image: { imageName: string } } }> } } };
    };

    expect(data.canvas.document.layers[0]?.source.image.imageName).toBe('board-shared.png');
  });

  /** The dedup that document references get is deliberately not applied to board membership. */
  it('copies board media even when this server already has that name', async () => {
    acceptCreate();
    transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));

    await projectFile.importProjectFile(await exportedBoardArchive());

    expect(uploadedBoardNames().sort()).toEqual(['shared.png', 'unreferenced.png']);
  });

  /**
   * The failure that must never bind to a stranger: on this server `shared.png` is taken, by the
   * project this archive came from.
   */
  it('forces an overlapping reference dangling when its board upload fails', async () => {
    acceptCreate();
    // The destination has an image called `shared.png` — it is the source project's own. Falling
    // back to that name would open the copy pointing at somebody else's picture.
    transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));
    transport.uploadBoardImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
      fileName === 'shared.png'
        ? Promise.reject(new Error('rejected'))
        : Promise.resolve({ height: 1, imageName: `board-${fileName}`, width: 1 })
    );

    const outcome = await projectFile.importProjectFile(await exportedBoardArchive());
    const { data } = api.createProject.mock.calls[0]![0] as {
      data: { canvas: { document: { layers: Array<{ source: { image: { imageName: string } } }> } } };
    };
    const restoredName = data.canvas.document.layers[0]!.source.image.imageName;

    expect(restoredName).not.toBe('shared.png');
    expect(restoredName).toContain('-missing-image-');
    expect(outcome.boardItemIssues).toEqual([{ kind: 'image', name: 'shared.png', reason: 'upload-failed' }]);
    expect(outcome.documentReferenceIssues).toEqual([{ kind: 'image', name: 'shared.png', reason: 'upload-failed' }]);
  });

  it('points the imported document at the board the server says it claimed', async () => {
    acceptCreate();

    const { record } = await projectFile.importProjectFile(await exportedBoardArchive());
    const values = Object.values(record.data.widgetInstances as Record<string, { state?: { values?: unknown } }>)
      .map((instance) => instance.state?.values as { projectBoardId?: string; selectedBoardId?: string } | undefined)
      .filter((instanceValues) => instanceValues?.projectBoardId !== undefined);

    expect(values.length).toBeGreaterThan(0);
    expect(values[0]).toMatchObject({ projectBoardId: 'staging-board', selectedBoardId: 'staging-board' });
  });

  it('creates no staging board for an archive whose board was empty', async () => {
    acceptCreate();
    await projectFile.exportOpenProject(boardProject());
    await projectFile.importProjectFile(capturedArchive());

    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(api.createProject.mock.calls[0]![0]).not.toHaveProperty('board_id');
  });

  it('creates no staging board for a document that will not rehydrate', async () => {
    const { textEntry, writeArchive } = await import('./invk/archive');
    const blob = await writeArchive(
      new Map([
        [
          'manifest.json',
          textEntry(
            JSON.stringify({
              appVersion: '7.0',
              contents: 'workbench-project',
              createdAt: '',
              name: 'No layout',
              version: 2,
            })
          ),
        ],
        ['board.json', textEntry(JSON.stringify({ items: [], version: 1 }))],
        ['project.json', textEntry(JSON.stringify({ name: 'No layout' }))],
      ])
    );

    await expect(projectFile.importProjectFile(new File([blob], 'broken.invk'))).rejects.toMatchObject({
      reason: 'damaged',
    });
    expect(transport.createStagingBoard).not.toHaveBeenCalled();
  });

  it('deletes the media it created and then the staging board when the create fails', async () => {
    const { ApiError } = await import('@platform/transport/http');
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new Error('project create rejected');

    account.accountLifecycle.activate('board-rollback-user');

    const archive = await exportedBoardArchive();

    api.createProject.mockRejectedValue(primaryFailure);
    api.getProject.mockRejectedValueOnce(new ApiError('not found', 404));

    await expect(projectFile.importProjectFile(archive)).rejects.toBe(primaryFailure);

    expect(transport.deleteArchiveImages).toHaveBeenCalledWith(
      ['board-shared.png', 'board-unreferenced.png'],
      expect.anything()
    );
    expect(transport.deleteArchiveVideos).toHaveBeenCalledWith(['board-clip.mp4'], expect.anything());
    expect(transport.deleteStagingBoard).toHaveBeenCalledWith('staging-board', expect.anything());
    account.accountLifecycle.invalidate();
  });

  it('leaves the staging board alone when a rejected create may already have claimed it', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new Error('connection ended after create');

    account.accountLifecycle.activate('board-ambiguous-user');

    const archive = await exportedBoardArchive();

    api.createProject.mockRejectedValue(primaryFailure);
    api.getProject.mockResolvedValue({
      board_id: 'staging-board',
      created_at: '2026-06-10 10:00:00.000',
      data: {},
      name: 'Board project',
      project_id: 'committed',
      revision: 1,
      updated_at: '2026-06-10 10:00:00.000',
    });

    await expect(projectFile.importProjectFile(archive)).rejects.toBe(primaryFailure);

    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});

describe('importProjectFile', () => {
  it('round-trips an exported project under a fresh id', async () => {
    const project = { ...createDraftProject([]), name: 'Exported project' };

    acceptCreate();
    await projectFile.exportOpenProject(project);

    const { record } = await projectFile.importProjectFile(capturedArchive());

    expect(record.project_id).not.toBe(project.id);
    expect(record.name).toBe('Exported project');

    const createRequest = api.createProject.mock.calls[0]![0] as { data: Record<string, unknown>; project_id: string };

    expect(createRequest.data.id).toBe(createRequest.project_id);
  });

  /**
   * Export strips installation state, so its own archives never carry it — but import must not
   * rely on that. A legacy `.invokeproject.json`, a dev-build archive or a hand-edited one can all
   * arrive with a stranger's gallery selection, and the collector skips those keys, so a restore
   * can neither fetch what they point at nor report it as dangling.
   */
  it('strips a stranger’s gallery selection from a document it did not write', async () => {
    acceptCreate();
    const envelope = {
      document: {
        ...createDraftProject([]),
        name: 'Handed over',
        widgetInstances: {
          'gallery-1': {
            state: {
              values: {
                compareImage: { imageName: 'theirs-compare.png', imageUrl: '' },
                selectedImage: { imageName: 'theirs-selected.png', kind: 'image' },
                selectedImageNames: ['image:theirs-selected.png'],
              },
            },
            typeId: 'gallery',
          },
        },
      },
      kind: 'invokeai-project',
      version: 1,
    };
    const file = new File([JSON.stringify(envelope)], 'handed-over.invokeproject.json');

    await projectFile.importProjectFile(file);

    const { data } = api.createProject.mock.calls[0]![0] as { data: Record<string, unknown> };
    const serialized = JSON.stringify(data);

    expect(serialized).not.toContain('theirs-compare.png');
    expect(serialized).not.toContain('theirs-selected.png');
  });

  it('uploads only the images the server is missing', async () => {
    const project = createDraftProject([]);

    acceptCreate();
    await projectFile.exportOpenProject(project);
    transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));

    await projectFile.importProjectFile(capturedArchive());

    expect(transport.uploadArchiveImage).not.toHaveBeenCalled();
  });

  it('refuses a legacy canvas project by reason', async () => {
    const { binaryEntry, textEntry, writeArchive } = await import('./invk/archive');
    const blob = await writeArchive(
      new Map([
        ['manifest.json', textEntry(JSON.stringify({ appVersion: '6.9', createdAt: '', name: 'C', version: 1 }))],
        ['canvas_state.json', binaryEntry(new Uint8Array([1]))],
      ])
    );

    await expect(projectFile.importProjectFile(new File([blob], 'legacy.invk'))).rejects.toMatchObject({
      reason: 'legacy-canvas-project',
    });
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('refuses a file that is not an archive before touching the server', async () => {
    await expect(projectFile.importProjectFile(new File(['{"some":"json"}'], 'x.invk'))).rejects.toMatchObject({
      reason: 'not-a-project',
    });
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('refuses a damaged document before any asset or project mutation', async () => {
    const { binaryEntry, textEntry, writeArchive } = await import('./invk/archive');
    const blob = await writeArchive(
      new Map([
        [
          'manifest.json',
          textEntry(
            JSON.stringify({
              appVersion: '7.0',
              contents: 'workbench-project',
              createdAt: '',
              name: 'No layout',
              version: 2,
            })
          ),
        ],
        [
          'project.json',
          textEntry(
            JSON.stringify({
              imageName: 'missing-layout.png',
              name: 'No layout',
              video_name: 'missing-layout.mp4',
            })
          ),
        ],
        ['images/missing-layout.png', binaryEntry(new Uint8Array([1]))],
        ['videos/missing-layout.mp4', binaryEntry(new Uint8Array([2]))],
      ])
    );

    await expect(projectFile.importProjectFile(new File([blob], 'broken.invk'))).rejects.toMatchObject({
      reason: 'damaged',
    });
    expect(transport.findExistingImageNames).not.toHaveBeenCalled();
    expect(transport.findExistingVideoNames).not.toHaveBeenCalled();
    expect(transport.uploadArchiveImage).not.toHaveBeenCalled();
    expect(transport.uploadArchiveVideo).not.toHaveBeenCalled();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(covers.recordProjectCover).not.toHaveBeenCalled();
  });

  it('canonicalizes a legacy document before restoring assets and persisting it', async () => {
    const { binaryEntry, textEntry, writeArchive } = await import('./invk/archive');
    const project = createDraftProject([]);
    const document = {
      ...persistence.serializeProjectDocument(project),
      canvas: {
        ...project.canvas,
        document: {
          ...project.canvas.document,
          layers: [rasterImageLayer('image-layer', 'legacy.png')],
        },
      },
      futureDocumentKey: { survives: true },
      invocation: { sourceId: 'project-graph' },
      name: ' Legacy project ',
    };
    const blob = await writeArchive(
      new Map([
        [
          'manifest.json',
          textEntry(
            JSON.stringify({
              appVersion: '7.0',
              contents: 'workbench-project',
              createdAt: '',
              name: 'Legacy project',
              version: 2,
            })
          ),
        ],
        ['project.json', textEntry(JSON.stringify(document))],
        ['images/legacy.png', binaryEntry(new Uint8Array([1]))],
      ])
    );

    acceptCreate();

    await projectFile.importProjectFile(new File([blob], 'legacy.invk'));

    const createRequest = api.createProject.mock.calls[0]![0] as { data: Record<string, unknown> };
    const invocation = createRequest.data.invocation as { sourceId: string };
    const canvas = createRequest.data.canvas as {
      document: { layers: Array<{ source: { image: { imageName: string } } }> };
    };

    expect(invocation.sourceId).toBe('workflow');
    expect(createRequest.data.futureDocumentKey).toEqual({ survives: true });
    expect(canvas.document.layers[0]?.source.image.imageName).toBe('server-legacy.png');
  });

  it('imports the shipped legacy JSON envelope under a fresh canonical identity without restoring assets', async () => {
    const legacyDocument = {
      ...persistence.serializeProjectDocument(projectWithRestorableAssets(false)),
      futureDocumentKey: { survives: true },
      id: 'legacy-project-id',
      invocation: { sourceId: 'project-graph' },
      name: ' Legacy JSON project ',
    };
    const file = new File(
      [
        JSON.stringify({
          document: legacyDocument,
          exportedAt: '2026-01-01T00:00:00.000Z',
          kind: 'invokeai-project',
          version: 1,
        }),
      ],
      'Legacy JSON project.invokeproject.json',
      { type: 'application/json' }
    );

    acceptCreate();

    const { record } = await projectFile.importProjectFile(file);
    const request = api.createProject.mock.calls[0]![0] as {
      data: Record<string, unknown>;
      name: string;
      project_id: string;
    };

    expect(record.name).toBe('Legacy JSON project');
    expect(request.project_id).not.toBe('legacy-project-id');
    expect(request.data.id).toBe(request.project_id);
    expect(request.data.invocation).toMatchObject({ sourceId: 'workflow' });
    expect(request.data.futureDocumentKey).toEqual({ survives: true });
    expect(transport.findExistingImageNames).not.toHaveBeenCalled();
    expect(transport.findExistingVideoNames).not.toHaveBeenCalled();
    expect(transport.uploadArchiveImage).not.toHaveBeenCalled();
    expect(transport.uploadArchiveVideo).not.toHaveBeenCalled();
  });

  it.each([
    ['malformed JSON', '{'],
    ['another product kind', JSON.stringify({ document: {}, kind: 'other', version: 1 })],
    [
      'an unsupported version',
      JSON.stringify({
        document: {},
        kind: 'invokeai-project',
        version: 2,
      }),
    ],
    ['a missing document', JSON.stringify({ kind: 'invokeai-project', version: 1 })],
  ])('refuses a legacy JSON envelope with %s before any mutation', async (_case, contents) => {
    const file = new File([contents], 'invalid.invokeproject.json', { type: 'application/json' });

    await expect(projectFile.importProjectFile(file)).rejects.toMatchObject({ reason: 'not-a-project' });
    expect(api.createProject).not.toHaveBeenCalled();
    expect(transport.findExistingImageNames).not.toHaveBeenCalled();
    expect(transport.findExistingVideoNames).not.toHaveBeenCalled();
    expect(transport.uploadArchiveImage).not.toHaveBeenCalled();
    expect(transport.uploadArchiveVideo).not.toHaveBeenCalled();
  });

  it('refuses an oversized legacy JSON file before materializing its text', async () => {
    const { INVK_MAX_ARCHIVE_BYTES } = await import('./invk/archive');
    const text = vi.fn(() => Promise.reject(new Error('legacy text was materialized')));
    const file = {
      name: 'oversized.invokeproject.json',
      size: INVK_MAX_ARCHIVE_BYTES + 1,
      text,
    } as unknown as File;

    await expect(projectFile.importProjectFile(file)).rejects.toMatchObject({ reason: 'too-large' });
    expect(text).not.toHaveBeenCalled();
  });

  it('rolls back every authoritative uploaded identity when project creation fails without hiding that failure', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const { ApiError } = await import('@platform/transport/http');
    const primaryFailure = new Error('project create rejected');

    account.accountLifecycle.activate('rollback-user');
    await projectFile.exportOpenProject(projectWithRestorableAssets());
    api.createProject.mockRejectedValue(primaryFailure);
    api.getProject.mockRejectedValueOnce(new ApiError('not found', 404));
    transport.deleteArchiveImages.mockRejectedValueOnce(new Error('image cleanup failed'));
    transport.deleteArchiveVideos.mockRejectedValueOnce(new Error('video cleanup failed'));

    await expect(projectFile.importProjectFile(capturedArchive())).rejects.toBe(primaryFailure);

    expect(transport.deleteArchiveImages).toHaveBeenCalledWith(['server-archive-image.png'], expect.anything());
    expect(transport.deleteArchiveVideos).toHaveBeenCalledWith(['server-archive-video.mp4'], expect.anything());
    account.accountLifecycle.invalidate();
  });

  it('does not roll back when a rejected create may already have committed the project', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new Error('connection ended after create');

    account.accountLifecycle.activate('ambiguous-response-user');
    await projectFile.exportOpenProject(projectWithRestorableAssets(false));
    api.createProject.mockRejectedValue(primaryFailure);
    api.getProject.mockResolvedValue({
      created_at: '2026-06-10 10:00:00.000',
      data: {},
      name: 'Committed project',
      project_id: 'committed-project',
      revision: 1,
      updated_at: '2026-06-10 10:00:00.000',
    });

    await expect(projectFile.importProjectFile(capturedArchive())).rejects.toBe(primaryFailure);

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteArchiveVideos).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });

  it('does not issue destructive rollback requests under a different account', async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('rollback-user-a');
    await projectFile.exportOpenProject(projectWithRestorableAssets(false));
    transport.uploadArchiveImage.mockImplementationOnce((_bytes: Uint8Array, fileName: string) => {
      account.accountLifecycle.activate('rollback-user-b');

      return Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 });
    });

    await expect(projectFile.importProjectFile(capturedArchive())).rejects.toThrow('no longer active');

    expect(api.createProject).not.toHaveBeenCalled();
    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteArchiveVideos).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });

  it('does not roll back assets after an ambiguous create that returned before the account changed', async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('ambiguous-create-user-a');
    await projectFile.exportOpenProject(projectWithRestorableAssets(false));
    api.createProject.mockImplementation((request: { name: string; project_id?: string }) => {
      account.accountLifecycle.activate('ambiguous-create-user-b');

      return Promise.resolve({
        created_at: '2026-06-10 10:00:00.000',
        data: {},
        name: request.name,
        project_id: request.project_id ?? '',
        revision: 1,
        updated_at: '2026-06-10 10:00:00.000',
      });
    });

    await expect(projectFile.importProjectFile(capturedArchive())).rejects.toThrow('no longer active');

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteArchiveVideos).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });

  it('does not upload an account A file after local parsing completes under account B', async () => {
    const account = await import('@platform/state/accountLifecycle');

    acceptCreate();
    await projectFile.exportOpenProject({ ...createDraftProject([]), name: 'Account A project' });

    const file = capturedArchive();
    const bytes = await file.arrayBuffer();
    const contents = deferred<ArrayBuffer>();

    vi.spyOn(file, 'arrayBuffer').mockReturnValue(contents.promise);
    account.accountLifecycle.activate('project-import-a');

    const imported = projectFile.importProjectFile(file);

    account.accountLifecycle.activate('project-import-b');
    contents.resolve(bytes);

    await expect(imported).rejects.toThrow('no longer active');
    expect(api.createProject).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});
