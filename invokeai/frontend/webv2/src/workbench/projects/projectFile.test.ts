import { createDraftProject } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

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
  setClientStateValue: vi.fn(() => Promise.resolve()),
}));

const downloads = vi.hoisted(() => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

const covers = vi.hoisted(() => ({ recordProjectCover: vi.fn() }));

const transport = vi.hoisted(() => ({
  coverExtensionForMime: () => 'webp',
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
  uploadArchiveImage: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
  ),
  uploadArchiveVideo: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ videoName: `server-${fileName}` })
  ),
}));

vi.mock('./api', () => api);
vi.mock('./covers', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./covers')>()),
  recordProjectCover: covers.recordProjectCover,
}));
vi.mock('@platform/browser/downloadBlob', () => downloads);
vi.mock('./invk/assetTransport', () => transport);

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

const capturedArchive = (): File => {
  const [blob, fileName] = downloads.downloadBlob.mock.calls.at(-1)! as [Blob, string];

  return new File([blob], fileName);
};

const acceptCreate = (): void => {
  api.createProject.mockImplementation((request: { name: string; project_id?: string }) =>
    Promise.resolve({
      created_at: '2026-06-10 10:00:00.000',
      data: {},
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
  transport.uploadArchiveImage.mockImplementation((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
  );
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

    expect(outcome.missingAssetNames).toEqual(['b.png']);
    expect(outcome.fileName).toBe('Two layers.invk');
  });

  it('reports nothing lost for a clean export', async () => {
    const document = { ...persistence.serializeProjectDocument(createDraftProject([])), ...projectWithImages() };

    api.getProject.mockResolvedValue({ data: document, name: 'Two layers', project_id: 'p1', revision: 1 });

    const outcome = await projectFile.exportLibraryProject('p1');

    expect(outcome.missingAssetNames).toEqual([]);
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

    expect(outcome.danglingAssetNames).toEqual(['b.png']);
    expect(onProgress.mock.calls.map(([progress]) => progress.phase)).toContain('restoring');
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
