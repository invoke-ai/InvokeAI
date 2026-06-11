import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createDraftProject } from '../workbenchState';
import type * as projectFileModule from './projectFile';
import type * as persistenceModule from './syncedPersistence';

/**
 * The portable project file format: versioned envelope, defensive parsing,
 * and the import rule that a file can never overwrite an existing project
 * because it always lands under a fresh id.
 */

const api = vi.hoisted(() => ({
  createProject: vi.fn(),
  getProject: vi.fn(),
}));

vi.mock('./api', () => api);

let projectFile: typeof projectFileModule;
let persistence: typeof persistenceModule;

beforeEach(async () => {
  vi.resetModules();
  api.createProject.mockReset();
  api.getProject.mockReset();

  projectFile = await import('./projectFile');
  persistence = await import('./syncedPersistence');
});

describe('project file envelope', () => {
  it('round-trips a serialized project document', () => {
    const project = createDraftProject([]);
    const document = persistence.serializeProjectDocument(project);
    const file = projectFile.buildProjectFile(document);
    const parsed = projectFile.parseProjectFile(JSON.stringify(file));

    expect(parsed).toEqual(document);
  });

  it('rejects files that are not project exports', () => {
    expect(projectFile.parseProjectFile('not json')).toBeNull();
    expect(projectFile.parseProjectFile('{}')).toBeNull();
    expect(projectFile.parseProjectFile(JSON.stringify({ document: {}, kind: 'other', version: 1 }))).toBeNull();
    expect(
      projectFile.parseProjectFile(JSON.stringify({ document: {}, kind: 'invokeai-project', version: 99 }))
    ).toBeNull();
    expect(projectFile.parseProjectFile(JSON.stringify({ kind: 'invokeai-project', version: 1 }))).toBeNull();
  });
});

describe('importProjectFile', () => {
  const fileFor = (contents: unknown): File => new File([JSON.stringify(contents)], 'export.invokeproject.json');

  it('creates the project under a fresh id, never the id in the file', async () => {
    const project = { ...createDraftProject([]), name: 'Exported project' };
    const document = persistence.serializeProjectDocument(project);

    api.createProject.mockImplementation((request: { project_id?: string; name: string }) =>
      Promise.resolve({
        created_at: '2026-06-10 10:00:00.000',
        data: {},
        name: request.name,
        project_id: request.project_id ?? '',
        revision: 1,
        updated_at: '2026-06-10 10:00:00.000',
      })
    );

    const record = await projectFile.importProjectFile(fileFor(projectFile.buildProjectFile(document)));

    expect(record.project_id).not.toBe(project.id);
    expect(record.name).toBe('Exported project');

    const createRequest = api.createProject.mock.calls[0][0] as { data: Record<string, unknown>; project_id: string };

    expect(createRequest.data.id).toBe(createRequest.project_id);
  });

  it('rejects files whose document is not a usable project', async () => {
    const file = fileFor(projectFile.buildProjectFile({ name: 'No layout' }));

    await expect(projectFile.importProjectFile(file)).rejects.toThrow('damaged');
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('rejects non-project files before touching the server', async () => {
    await expect(projectFile.importProjectFile(fileFor({ some: 'json' }))).rejects.toThrow('not an Invoke project');
    expect(api.createProject).not.toHaveBeenCalled();
  });
});
