import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryModule from './library';

/**
 * The project library store: summaries normalized at the boundary, sorted by
 * recency, and the explicit mutations (delete, rename, duplicate) that are
 * the only paths off the server.
 */

const api = vi.hoisted(() => ({
  createProject: vi.fn(),
  deleteProject: vi.fn(),
  getProject: vi.fn(),
  listProjects: vi.fn(),
  updateProject: vi.fn(),
}));

vi.mock('./api', () => api);

let library: typeof libraryModule;

const summaryDto = (id: string, name: string, updatedAt: string) => ({
  created_at: '2026-06-01 08:00:00.000',
  name,
  project_id: id,
  revision: 1,
  updated_at: updatedAt,
});

beforeEach(async () => {
  vi.resetModules();
  api.createProject.mockReset();
  api.deleteProject.mockReset();
  api.getProject.mockReset();
  api.listProjects.mockReset();
  api.updateProject.mockReset();

  library = await import('./library');
});

describe('refreshProjectLibrary', () => {
  it('normalizes SQLite timestamps to ISO and sorts newest first', async () => {
    api.listProjects.mockResolvedValue([
      summaryDto('older', 'Older', '2026-06-09 10:00:00.000'),
      summaryDto('newer', 'Newer', '2026-06-10 10:00:00.000'),
    ]);

    await library.refreshProjectLibrary();

    const { status, summaries } = library.getProjectLibrary();

    expect(status).toBe('ready');
    expect(summaries.map((summary) => summary.id)).toEqual(['newer', 'older']);
    expect(summaries[0].updatedAt).toBe('2026-06-10T10:00:00.000Z');
  });

  it('keeps the previous summaries and reports the failure on error', async () => {
    api.listProjects.mockResolvedValue([summaryDto('kept', 'Kept', '2026-06-10 10:00:00.000')]);
    await library.refreshProjectLibrary();

    api.listProjects.mockRejectedValue(new Error('offline'));
    await library.refreshProjectLibrary();

    const { error, status, summaries } = library.getProjectLibrary();

    expect(status).toBe('error');
    expect(error).toBe('offline');
    expect(summaries.map((summary) => summary.id)).toEqual(['kept']);
  });
});

describe('upsertProjectSummary', () => {
  it('inserts new entries and moves updated ones to the front', async () => {
    api.listProjects.mockResolvedValue([
      summaryDto('a', 'A', '2026-06-09 10:00:00.000'),
      summaryDto('b', 'B', '2026-06-10 10:00:00.000'),
    ]);
    await library.refreshProjectLibrary();

    library.upsertProjectSummary({ id: 'a', name: 'A renamed', revision: 2 });

    const { summaries } = library.getProjectLibrary();

    expect(summaries[0].id).toBe('a');
    expect(summaries[0].name).toBe('A renamed');
    expect(summaries[0].revision).toBe(2);
  });
});

describe('library mutations', () => {
  it('deleteLibraryProject removes from server and store', async () => {
    api.listProjects.mockResolvedValue([summaryDto('doomed', 'Doomed', '2026-06-10 10:00:00.000')]);
    await library.refreshProjectLibrary();
    api.deleteProject.mockResolvedValue(undefined);

    await library.deleteLibraryProject('doomed');

    expect(api.deleteProject).toHaveBeenCalledWith('doomed');
    expect(library.getProjectLibrary().summaries).toHaveLength(0);
  });

  it('renameLibraryProject updates name in both the record and its document', async () => {
    api.getProject.mockResolvedValue({
      ...summaryDto('p1', 'Old name', '2026-06-10 10:00:00.000'),
      data: { id: 'p1', layout: {}, name: 'Old name' },
    });
    api.updateProject.mockResolvedValue({
      ...summaryDto('p1', 'New name', '2026-06-10 11:00:00.000'),
      data: { id: 'p1', layout: {}, name: 'New name' },
      revision: 2,
    });

    await library.renameLibraryProject('p1', 'New name');

    expect(api.updateProject).toHaveBeenCalledWith('p1', {
      data: { id: 'p1', layout: {}, name: 'New name' },
      expected_revision: 1,
      name: 'New name',
    });
    expect(library.getProjectLibrary().summaries[0]?.name).toBe('New name');
  });

  it('duplicateLibraryProject copies under a fresh id', async () => {
    api.getProject.mockResolvedValue({
      ...summaryDto('source', 'Source', '2026-06-10 10:00:00.000'),
      data: { id: 'source', layout: {}, name: 'Source' },
    });
    api.createProject.mockImplementation((request: { project_id?: string; name: string }) =>
      Promise.resolve({ ...summaryDto(request.project_id ?? '', request.name, '2026-06-10 11:00:00.000'), data: {} })
    );

    const copy = await library.duplicateLibraryProject('source');

    expect(copy.id).not.toBe('source');
    expect(copy.name).toBe('Source copy');

    const createRequest = api.createProject.mock.calls[0][0] as { data: Record<string, unknown>; project_id: string };

    expect(createRequest.data.id).toBe(createRequest.project_id);
    expect(createRequest.data.name).toBe('Source copy');
  });
});
