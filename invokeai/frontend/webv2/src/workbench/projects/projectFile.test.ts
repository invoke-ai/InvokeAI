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

const transport = vi.hoisted(() => ({
  coverExtensionForMime: () => 'webp',
  fetchImageBytes: vi.fn((imageName: string) => Promise.resolve(new TextEncoder().encode(`bytes:${imageName}`))),
  fetchImageThumbnail: vi.fn(() => Promise.resolve(null)),
  findExistingImageNames: vi.fn((_names: readonly string[]) => Promise.resolve(new Set<string>())),
  mimeForEntryName: () => 'image/png',
  uploadArchiveImage: vi.fn((_bytes: Uint8Array, fileName: string) =>
    Promise.resolve({ height: 1, imageName: `server-${fileName}`, width: 1 })
  ),
}));

vi.mock('./api', () => api);
vi.mock('@platform/browser/downloadBlob', () => downloads);
vi.mock('./invk/imageTransport', () => transport);

let projectFile: typeof projectFileModule;
let persistence: typeof persistenceModule;

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

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

describe('importProjectFile', () => {
  it('round-trips an exported project under a fresh id', async () => {
    const project = { ...createDraftProject([]), name: 'Exported project' };

    acceptCreate();
    await projectFile.exportOpenProject(project);

    const record = await projectFile.importProjectFile(capturedArchive());

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

  it('refuses an archive whose document is not a usable project', async () => {
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
        ['project.json', textEntry(JSON.stringify({ name: 'No layout' }))],
      ])
    );

    await expect(projectFile.importProjectFile(new File([blob], 'broken.invk'))).rejects.toMatchObject({
      reason: 'damaged',
    });
    expect(api.createProject).not.toHaveBeenCalled();
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
