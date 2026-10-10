import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDraftProject } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectBoardItemDTO, ProjectBoardSnapshotDTO } from './api';
import type * as apiModule from './api';
import type * as coversModule from './covers';
import type * as assetTransportModule from './invk/assetTransport';
import type * as projectFileModule from './projectFile';
import type { ProjectPushOutcome } from './projectFlush';
import type * as persistenceModule from './syncedPersistence';

import { ProjectCreateAbsentError } from './api';

/** Verify fresh IDs, no overwrites, and format rejection before server mutation. */

const api = vi.hoisted(() => ({
  createProjectSettled: vi.fn(),
  deleteClientStateValue: vi.fn(() => Promise.resolve()),
  getClientStateValue: vi.fn(() => Promise.resolve(null)),
  getProject: vi.fn(),
  // Every export enumerates the project's board; most of these cases do not care what is on it.
  getProjectBoardSnapshot: vi.fn((): Promise<ProjectBoardSnapshotDTO> =>
    Promise.resolve({ boards: [{ archived: false, board_id: 'inbox', is_inbox: true, items: [], name: 'Project' }] })
  ),
  isProjectNotFoundError: (error: unknown) =>
    typeof error === 'object' && error !== null && 'status' in error && error.status === 404,
  setClientStateValue: vi.fn(() => Promise.resolve()),
}));

const downloads = vi.hoisted(() => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

const covers = vi.hoisted(() => ({ recordProjectCover: vi.fn() }));
const fontTransport = vi.hoisted(() => ({
  download: vi.fn(),
  remove: vi.fn(async () => {}),
  upload: vi.fn(),
  validate: vi.fn(async () => {}),
}));
vi.mock('./invk/fontTransport', () => ({ createFontArchiveTransport: () => fontTransport }));

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
  placeBoardInProject: vi.fn(() => Promise.resolve()),
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

vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof apiModule>()),
  ...api,
}));
vi.mock('./covers', async (importOriginal) => ({
  ...(await importOriginal<typeof coversModule>()),
  recordProjectCover: covers.recordProjectCover,
}));
vi.mock('@platform/browser/downloadBlob', () => downloads);
// Preserve real cancellation predicates so cancellation cannot masquerade as missing assets.
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
        stacks: stacksFrom([rasterImageLayer('restored-image', 'archive-image.png')]),
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
  api.createProjectSettled.mockImplementation(
    (request: {
      board_id?: string;
      data?: Record<string, unknown>;
      minimum_canvas_schema_version?: number;
      name: string;
      project_id?: string;
    }) =>
      Promise.resolve({
        board_id: request.board_id ?? 'server-created-board',
        created_at: '2026-06-10 10:00:00.000',
        data: request.data ?? {},
        minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 2,
        name: request.name,
        project_id: request.project_id ?? '',
        revision: 1,
        updated_at: '2026-06-10 10:00:00.000',
      })
  );
};

beforeEach(async () => {
  vi.resetModules();
  vi.resetAllMocks();

  projectFile = await import('./projectFile');
  persistence = await import('./syncedPersistence');
});

describe('exportOpenProject', () => {
  it('downloads an archive named after the project', async () => {
    await projectFile.exportOpenProject({ ...createDraftProject([]), name: 'My project' });

    expect(downloads.downloadBlob).toHaveBeenCalledTimes(1);
    expect(downloads.downloadBlob.mock.calls[0]![1]).toBe('My project.invk');
  });

  it("writes an open project's document after its canvas pixels reach it", async () => {
    const { registerOpenProject, unregisterOpenProject } = await import('./syncStore');
    const requested = { ...createDraftProject([]), name: 'Before barrier' };
    const order: string[] = [];
    api.getProject.mockResolvedValue({
      data: persistence.serializeProjectDocument(requested),
      name: requested.name,
      project_id: requested.id,
      revision: 1,
    });
    registerOpenProject(requested.id, {
      close: vi.fn(),
      current: vi.fn(() => {
        order.push('current');
        return { ...requested, name: 'After barrier' };
      }),
      deleteOnServer: vi.fn(),
      flush: vi.fn(() => Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'acknowledged' })),
      flushPixels: vi.fn(() => {
        order.push('flushPixels');
        return Promise.resolve();
      }),
      markDeleted: vi.fn(),
      rename: vi.fn(),
      unmarkDeleted: vi.fn(),
    });

    try {
      await projectFile.exportOpenProject(requested);
    } finally {
      unregisterOpenProject(requested.id);
    }

    expect(order.slice(0, 2)).toEqual(['flushPixels', 'current']);
    expect(downloads.downloadBlob.mock.calls[0]![1]).toBe('After barrier.invk');
  });

  /** Include a positive transport probe so negative server-call assertions cannot pass vacuously. */
  it('reaches the server through the mocked transport', async () => {
    api.getProject.mockResolvedValue({
      data: {
        canvas: {
          document: {
            stacks: { raster: [{ id: 'l', source: { image: { imageName: 'pinned.png' }, type: 'image' } }] },
          },
        },
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

  it('canonicalizes a supported legacy server record before exporting it', async () => {
    const document = {
      ...persistence.serializeProjectDocument({ ...createDraftProject([]), name: 'Legacy' }),
      events: [{ id: 'legacy-event' }],
      graphHistory: [{ id: 'legacy-graph' }],
      queue: { items: [{ id: 'legacy-queue' }] },
    };

    api.getProject.mockResolvedValue({ data: document, name: 'Legacy', project_id: 'p1', revision: 3 });

    await projectFile.exportLibraryProject('p1');

    const { readArchive, readEntryText } = await import('./invk/archive');
    const [blob] = downloads.downloadBlob.mock.calls.at(-1)! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));
    const exported = JSON.parse(readEntryText(entries.get('project.json')!)) as Record<string, unknown>;

    expect(exported.documentSchemaVersion).toBe(3);
    expect(exported).not.toHaveProperty('events');
    expect(exported).not.toHaveProperty('graphHistory');
    expect(exported).not.toHaveProperty('queue');
  });

  it('exports a future-schema server record verbatim for recovery', async () => {
    const document = { documentSchemaVersion: 4, futureField: { value: 42 }, id: 'future', name: 'Future' };

    api.getProject.mockResolvedValue({ data: document, name: 'Future', project_id: 'p1', revision: 3 });

    await projectFile.exportLibraryProject('p1');

    const { readArchive, readEntryText } = await import('./invk/archive');
    const [blob] = downloads.downloadBlob.mock.calls.at(-1)! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect(JSON.parse(readEntryText(entries.get('project.json')!))).toEqual(document);
  });

  it('records the source project compatibility floor in the archive', async () => {
    const document = persistence.serializeProjectDocument({ ...createDraftProject([]), name: 'Future history' });

    api.getProject.mockResolvedValue({
      data: document,
      minimum_canvas_schema_version: 4,
      name: 'Future history',
      project_id: 'p1',
      revision: 3,
    });

    await projectFile.exportLibraryProject('p1');

    const { readArchive, readEntryText } = await import('./invk/archive');
    const [blob] = downloads.downloadBlob.mock.calls.at(-1)! as [Blob];
    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));
    const manifest = JSON.parse(readEntryText(entries.get('manifest.json')!)) as Record<string, unknown>;

    expect(manifest.minimumCanvasSchemaVersion).toBe(4);
  });

  /** Flush open projects before export to include unacknowledged edits. */
  it('flushes an open project before reading its record', async () => {
    const { registerOpenProject, unregisterOpenProject } = await import('./syncStore');
    const order: string[] = [];
    const flush = vi.fn(() => {
      order.push('flush');
      return Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'acknowledged' });
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
      deleteOnServer: vi.fn(),
      flush,
      current: vi.fn(() => undefined),
      flushPixels: vi.fn(() => {
        order.push('flushPixels');
        return Promise.resolve();
      }),
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
    expect(order).toEqual(['flushPixels', 'flush', 'get']);
  });

  /** Require an acknowledged flush before reading/exporting server bytes. */
  it('refuses to export a project whose flush never reached the server', async () => {
    const { registerOpenProject, unregisterOpenProject } = await import('./syncStore');

    registerOpenProject('p1', {
      close: vi.fn(),
      deleteOnServer: vi.fn(),
      flush: vi.fn(() => Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'unsynced' })),
      current: vi.fn(() => undefined),
      flushPixels: vi.fn(() => Promise.resolve()),
      markDeleted: vi.fn(),
      rename: vi.fn(),
      unmarkDeleted: vi.fn(),
    });

    try {
      await expect(projectFile.exportLibraryProject('p1')).rejects.toMatchObject({
        name: 'ProjectFlushError',
        reason: 'unsynced',
      });
    } finally {
      unregisterOpenProject('p1');
    }

    expect(api.getProject).not.toHaveBeenCalled();
    expect(downloads.downloadBlob).not.toHaveBeenCalled();
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

/** Partial-loss outcomes must reach callers. */
describe('what a transfer reports', () => {
  const projectWithImages = () => {
    const project = createDraftProject([]);

    return {
      ...project,
      canvas: {
        ...project.canvas,
        document: {
          ...project.canvas.document,
          stacks: stacksFrom([rasterImageLayer('l1', 'a.png'), rasterImageLayer('l2', 'b.png')]),
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

/** Copy imported board media to fresh identities; adopting existing names would move another board's media. */
describe('importing a project board', () => {
  const boardProject = () => {
    const project = createDraftProject([]);

    return {
      ...project,
      canvas: {
        ...project.canvas,
        document: { ...project.canvas.document, stacks: stacksFrom([rasterImageLayer('l1', 'shared.png')]) },
      },
      name: 'Board project',
    };
  };

  const inboxItems: ProjectBoardItemDTO[] = [
    { category: 'general', kind: 'image', name: 'shared.png', starred: true },
    { category: 'user', kind: 'image', name: 'unreferenced.png', starred: false },
    { category: 'general', kind: 'video', name: 'clip.mp4', starred: false },
  ];
  const boardSnapshot = (): ProjectBoardSnapshotDTO => ({
    boards: [{ archived: false, board_id: 'source-inbox', is_inbox: true, items: inboxItems, name: 'Board project' }],
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
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board' });
    expect(outcome.boardItemIssues).toEqual([]);
    expect(outcome.documentReferenceIssues).toEqual([]);
  });

  it('drops library links on import: a portable file never acquires a write target', async () => {
    acceptCreate();
    api.getProjectBoardSnapshot.mockResolvedValue(boardSnapshot());
    const project = boardProject();
    await projectFile.exportOpenProject({
      ...project,
      workflows: {
        ...project.workflows,
        entries: project.workflows.entries.map((entry) => ({
          ...entry,
          source: { libraryWorkflowId: 'lib-1', revision: 2 },
        })),
      },
    });

    await projectFile.importProjectFile(capturedArchive());

    const { data } = api.createProjectSettled.mock.calls[0]![0] as {
      data: { workflows: { entries: { document: { id: string }; source?: unknown }[] } };
    };

    expect(data.workflows.entries).toHaveLength(1);
    expect(data.workflows.entries[0]).not.toHaveProperty('source');
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

    const { data } = api.createProjectSettled.mock.calls[0]![0] as {
      data: { canvas: { document: { stacks: { raster: Array<{ source: { image: { imageName: string } } }> } } } };
    };

    expect(data.canvas.document.stacks.raster[0]?.source.image.imageName).toBe('board-shared.png');
  });

  /** The dedup that document references get is deliberately not applied to board membership. */
  it('copies board media even when this server already has that name', async () => {
    acceptCreate();
    transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));

    await projectFile.importProjectFile(await exportedBoardArchive());

    expect(uploadedBoardNames().sort()).toEqual(['shared.png', 'unreferenced.png']);
  });

  it('forces an overlapping reference dangling when its board upload fails', async () => {
    acceptCreate();
    // Failed copies must not reuse existing source-owned names.
    transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));
    transport.uploadBoardImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
      fileName === 'shared.png'
        ? Promise.reject(new Error('rejected'))
        : Promise.resolve({ height: 1, imageName: `board-${fileName}`, width: 1 })
    );

    const outcome = await projectFile.importProjectFile(await exportedBoardArchive());
    const { data } = api.createProjectSettled.mock.calls[0]![0] as {
      data: { canvas: { document: { stacks: { raster: Array<{ source: { image: { imageName: string } } }> } } } };
    };
    const restoredName = data.canvas.document.stacks.raster[0]!.source.image.imageName;

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

  it('stages and claims the inbox of an archive whose board was empty, uploading nothing', async () => {
    acceptCreate();
    await projectFile.exportOpenProject(boardProject());
    await projectFile.importProjectFile(capturedArchive());

    expect(transport.createStagingBoard).toHaveBeenCalledTimes(1);
    expect(transport.uploadBoardImage).not.toHaveBeenCalled();
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({
      board_id: 'staging-board',
      minimum_canvas_schema_version: 3,
    });
  });

  it('refuses an incompatible archive before staging or restoring any media', async () => {
    const { binaryEntry, readArchive, readEntryText, textEntry, writeArchive } = await import('./invk/archive');

    await projectFile.exportOpenProject(boardProject());

    const original = capturedArchive();
    const entries = await readArchive(new Uint8Array(await original.arrayBuffer()));
    const manifest = JSON.parse(readEntryText(entries.get('manifest.json')!)) as Record<string, unknown>;

    const rewrittenEntries = new Map([...entries].map(([path, bytes]) => [path, binaryEntry(bytes)]));

    rewrittenEntries.set('manifest.json', textEntry(JSON.stringify({ ...manifest, minimumCanvasSchemaVersion: 5 })));
    const archive = new File([await writeArchive(rewrittenEntries)], original.name);

    await expect(projectFile.importProjectFile(archive)).rejects.toMatchObject({ reason: 'unsupported-version' });

    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(transport.uploadBoardImage).not.toHaveBeenCalled();
    expect(transport.uploadBoardVideo).not.toHaveBeenCalled();
    expect(api.createProjectSettled).not.toHaveBeenCalled();
  });

  it('deletes the media it created and then the staging board when the create fails', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new ProjectCreateAbsentError(new Error('project create rejected'));

    account.accountLifecycle.activate('board-rollback-user');

    const archive = await exportedBoardArchive();

    api.createProjectSettled.mockRejectedValue(primaryFailure);

    await expect(projectFile.importProjectFile(archive)).rejects.toBe(primaryFailure);

    expect(transport.deleteArchiveImages).toHaveBeenCalledWith(
      ['board-shared.png', 'board-unreferenced.png'],
      expect.anything()
    );
    expect(transport.deleteArchiveVideos).toHaveBeenCalledWith(['board-clip.mp4'], expect.anything());
    expect(transport.deleteStagingBoard).toHaveBeenCalledWith('staging-board', expect.anything());
    account.accountLifecycle.invalidate();
  });

  it('leaves the staging board alone when a rejected create may already have landed', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new Error('connection ended after create');

    account.accountLifecycle.activate('board-ambiguous-user');

    const archive = await exportedBoardArchive();

    api.createProjectSettled.mockRejectedValue(primaryFailure);

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

    const createRequest = api.createProjectSettled.mock.calls[0]![0] as {
      data: Record<string, unknown>;
      project_id: string;
    };

    expect(createRequest.data.id).toBe(createRequest.project_id);
  });

  /** Strip installation state on import too; skipped references cannot be remapped or reported. */
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

    const { data } = api.createProjectSettled.mock.calls[0]![0] as { data: Record<string, unknown> };
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
    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(transport.findExistingImageNames).not.toHaveBeenCalled();
    expect(transport.findExistingVideoNames).not.toHaveBeenCalled();
    expect(transport.uploadArchiveImage).not.toHaveBeenCalled();
    expect(transport.uploadArchiveVideo).not.toHaveBeenCalled();
    expect(api.createProjectSettled).not.toHaveBeenCalled();
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
          stacks: stacksFrom([rasterImageLayer('image-layer', 'legacy.png')]),
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

    const createRequest = api.createProjectSettled.mock.calls[0]![0] as { data: Record<string, unknown> };
    const invocation = createRequest.data.invocation as { sourceId: string };
    const canvas = createRequest.data.canvas as {
      document: { stacks: { raster: Array<{ source: { image: { imageName: string } } }> } };
    };

    expect(invocation.sourceId).toBe('workflow');
    expect(createRequest.data.futureDocumentKey).toEqual({ survives: true });
    expect(canvas.document.stacks.raster[0]?.source.image.imageName).toBe('server-legacy.png');
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
    const request = api.createProjectSettled.mock.calls[0]![0] as {
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
        version: 4,
      }),
    ],
    ['a missing document', JSON.stringify({ kind: 'invokeai-project', version: 1 })],
  ])('refuses a legacy JSON envelope with %s before any mutation', async (_case, contents) => {
    const file = new File([contents], 'invalid.invokeproject.json', { type: 'application/json' });

    await expect(projectFile.importProjectFile(file)).rejects.toMatchObject({ reason: 'not-a-project' });
    expect(api.createProjectSettled).not.toHaveBeenCalled();
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
    const primaryFailure = new ProjectCreateAbsentError(new Error('project create rejected'));

    account.accountLifecycle.activate('rollback-user');
    await projectFile.exportOpenProject(projectWithRestorableAssets());
    api.createProjectSettled.mockRejectedValue(primaryFailure);
    transport.deleteArchiveImages.mockRejectedValueOnce(new Error('image cleanup failed'));
    transport.deleteArchiveVideos.mockRejectedValueOnce(new Error('video cleanup failed'));

    await expect(projectFile.importProjectFile(capturedArchive())).rejects.toBe(primaryFailure);

    expect(transport.deleteArchiveImages).toHaveBeenCalledWith(['server-archive-image.png'], expect.anything());
    expect(transport.deleteArchiveVideos).toHaveBeenCalledWith(['server-archive-video.mp4'], expect.anything());
    account.accountLifecycle.invalidate();
  });

  it('does not roll back when a rejected create is not proof the project is absent', async () => {
    const account = await import('@platform/state/accountLifecycle');
    const primaryFailure = new Error('connection ended after create');

    account.accountLifecycle.activate('ambiguous-response-user');
    await projectFile.exportOpenProject(projectWithRestorableAssets(false));
    api.createProjectSettled.mockRejectedValue(primaryFailure);

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

    expect(api.createProjectSettled).not.toHaveBeenCalled();
    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteArchiveVideos).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });

  it('does not roll back assets after an ambiguous create that returned before the account changed', async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('ambiguous-create-user-a');
    await projectFile.exportOpenProject(projectWithRestorableAssets(false));
    api.createProjectSettled.mockImplementation((request: { name: string; project_id?: string }) => {
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
    expect(api.createProjectSettled).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});

describe('embedded font project imports', () => {
  const fontProject = async () => {
    const { sha256Hex } = await import('@platform/browser/sha256');
    const bytes = new Uint8Array([0, 1, 0, 0, 4, 5, 6]);
    const font = {
      contentHash: await sha256Hex(bytes),
      family: 'Example',
      id: 'source-font',
      label: 'Example Regular',
    };
    const project = createDraftProject([]);
    const layer = {
      ...rasterImageLayer('text', 'unused.png'),
      source: {
        align: 'left' as const,
        color: '#ffffff',
        content: 'Portable typography',
        fontFamily: 'Example',
        fontRef: font,
        fontSize: 40,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text' as const,
      },
    };
    project.canvas.document.stacks = stacksFrom([layer]);
    fontTransport.download.mockResolvedValue({ bytes, filename: 'Example.ttf' });
    fontTransport.upload.mockResolvedValue({ created: true, font: { ...font, id: 'imported-font' } });
    await projectFile.exportOpenProject(project, { includeFonts: true });
    return capturedArchive();
  };

  it('validates included fonts, installs them privately, and persists remapped references', async () => {
    const archive = await fontProject();
    acceptCreate();
    const result = await projectFile.importProjectFile(archive);
    const { collectFontDependencies } = await import('./invk/fonts');
    expect(collectFontDependencies(result.record.data)[0]?.references).toEqual(['imported-font']);
    expect(fontTransport.validate).toHaveBeenCalledTimes(1);
    expect(fontTransport.upload).toHaveBeenCalledTimes(1);
    expect(fontTransport.remove).not.toHaveBeenCalled();
  });

  it('rejects invalid font payloads before creating any server resources', async () => {
    const archive = await fontProject();
    fontTransport.validate.mockRejectedValue(new Error('Invalid font tables'));
    await expect(projectFile.importProjectFile(archive)).rejects.toThrow('Invalid font tables');
    expect(fontTransport.upload).not.toHaveBeenCalled();
    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(api.createProjectSettled).not.toHaveBeenCalled();
  });

  it('cleans up newly installed fonts only when failed project creation proves absence', async () => {
    const archive = await fontProject();
    const { ProjectCreateAbsentError: AbsentError } = await import('./api');
    api.createProjectSettled.mockRejectedValue(new AbsentError(new Error('rejected')));
    await expect(projectFile.importProjectFile(archive)).rejects.toBeInstanceOf(AbsentError);
    expect(fontTransport.remove).toHaveBeenCalledWith('imported-font', expect.any(AbortSignal));
    fontTransport.remove.mockClear();
    api.createProjectSettled.mockRejectedValue(new Error('Unknown create outcome'));
    await expect(projectFile.importProjectFile(archive)).rejects.toThrow('Unknown create outcome');
    expect(fontTransport.remove).not.toHaveBeenCalled();
  });

  it('allows an explicit references-only retry without installing or deleting fonts', async () => {
    const archive = await fontProject();
    acceptCreate();
    const result = await projectFile.importProjectFile(archive, { skipEmbeddedFonts: true });
    const { collectFontDependencies } = await import('./invk/fonts');
    expect(collectFontDependencies(result.record.data)[0]?.references).toEqual(['source-font']);
    expect(fontTransport.upload).not.toHaveBeenCalled();
    expect(fontTransport.remove).not.toHaveBeenCalled();
  });
});

describe('importing every board of a project', () => {
  const boardProject = () => {
    const project = createDraftProject([]);

    return {
      ...project,
      canvas: {
        ...project.canvas,
        document: { ...project.canvas.document, stacks: stacksFrom([rasterImageLayer('l1', 'shared.png')]) },
      },
      name: 'Board project',
    };
  };
  const multiBoardSnapshot = (): ProjectBoardSnapshotDTO => ({
    boards: [
      {
        archived: false,
        board_id: 'source-inbox',
        is_inbox: true,
        items: [{ category: 'general', kind: 'image', name: 'shared.png', starred: true }],
        name: 'Board project',
      },
      {
        archived: true,
        board_id: 'source-old',
        is_inbox: false,
        items: [{ category: 'general', kind: 'image', name: 'old.png', starred: false }],
        name: 'Old façades',
      },
      { archived: false, board_id: 'source-empty', is_inbox: false, items: [], name: 'Site plan refs' },
    ],
  });
  const exportedArchive = async (): Promise<File> => {
    acceptCreate();
    api.getProjectBoardSnapshot.mockResolvedValue(multiBoardSnapshot());
    await projectFile.exportOpenProject(boardProject());

    return capturedArchive();
  };

  it('stages a board per archive board, claims the inbox on create, then moves the rest in', async () => {
    const archive = await exportedArchive();
    transport.createStagingBoard
      .mockResolvedValueOnce('staging-inbox')
      .mockResolvedValueOnce('staging-old')
      .mockResolvedValueOnce('staging-empty');

    const outcome = await projectFile.importProjectFile(archive);

    // Named for the user: the inbox after the project, the others after themselves, in the archive's order.
    expect((transport.createStagingBoard.mock.calls as unknown[][]).map((call) => call[0])).toEqual([
      'Board project',
      'Old façades',
      'Site plan refs',
    ]);
    expect(
      (transport.uploadBoardImage.mock.calls as unknown[][]).map((call) => [
        call[1],
        (call[2] as { boardId?: string } | undefined)?.boardId,
      ])
    ).toEqual([
      ['shared.png', 'staging-inbox'],
      ['old.png', 'staging-old'],
    ]);
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-inbox' });
    // Only once the project exists: the staged members move in, archived as they were.
    const projectId = outcome.record.project_id;
    expect(transport.placeBoardInProject.mock.calls).toEqual([
      ['staging-old', projectId, true, expect.anything()],
      ['staging-empty', projectId, false, expect.anything()],
    ]);
    expect(outcome.boardIssues).toEqual([]);
    expect(outcome.boardItemIssues).toEqual([]);
  });

  it('reports a board it could not place and keeps the imported project', async () => {
    const archive = await exportedArchive();
    transport.createStagingBoard.mockResolvedValueOnce('staging-inbox').mockResolvedValueOnce('staging-old');
    transport.placeBoardInProject.mockRejectedValueOnce(new Error('move refused'));

    const outcome = await projectFile.importProjectFile(archive);

    expect(outcome.boardIssues).toEqual([{ name: 'Old façades' }]);
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
  });

  it('deletes every staging board when the create provably did not happen', async () => {
    const archive = await exportedArchive();
    transport.createStagingBoard
      .mockResolvedValueOnce('staging-inbox')
      .mockResolvedValueOnce('staging-old')
      .mockResolvedValueOnce('staging-empty');
    api.createProjectSettled.mockRejectedValueOnce(new ProjectCreateAbsentError(new Error('refused')));

    await expect(projectFile.importProjectFile(archive)).rejects.toBeInstanceOf(ProjectCreateAbsentError);

    expect((transport.deleteStagingBoard.mock.calls as unknown[][]).map((call) => call[0]).sort()).toEqual([
      'staging-empty',
      'staging-inbox',
      'staging-old',
    ]);
    expect(transport.placeBoardInProject).not.toHaveBeenCalled();
  });

  it('deletes the boards it had staged when staging a later one fails, before any upload', async () => {
    const archive = await exportedArchive();
    transport.createStagingBoard.mockResolvedValueOnce('staging-inbox').mockRejectedValueOnce(new Error('refused'));

    await expect(projectFile.importProjectFile(archive)).rejects.toThrow('refused');

    expect(transport.uploadBoardImage).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).toHaveBeenCalledExactlyOnceWith('staging-inbox', expect.anything());
    expect(api.createProjectSettled).not.toHaveBeenCalled();
  });
});
