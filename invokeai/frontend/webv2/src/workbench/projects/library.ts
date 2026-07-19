import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';

import {
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  getProject as apiGetProject,
  listProjects,
  updateProject as apiUpdateProject,
  type ProjectSummaryDTO,
} from './api';
import { createProjectId } from './ids';

/**
 * The project library: every project saved on the server for the current
 * user, as lightweight summaries (no documents). The Home screen, the Open
 * Project dialog, and the editor's save pass all read and patch this one
 * store, so "what projects exist" has a single answer everywhere.
 *
 * The library is intentionally separate from the workbench session: open tabs
 * hold hydrated documents in workbench state, while the library only knows
 * metadata. Deleting through the library is the only way a project leaves the
 * server — closing a tab never does.
 */

export interface ProjectSummary {
  id: string;
  name: string;
  revision: number;
  createdAt: string;
  updatedAt: string;
}

export type ProjectLibraryStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface ProjectLibrarySnapshot {
  status: ProjectLibraryStatus;
  summaries: ProjectSummary[];
  error?: string;
}

const store = createExternalStore<ProjectLibrarySnapshot>({ status: 'idle', summaries: [] });

/**
 * The backend stamps timestamps via SQLite ("2026-06-11 09:21:04.123") —
 * UTC, but with no timezone marker, which `Date` would misread as local
 * time. Normalize to ISO once, here at the boundary.
 */
const normalizeServerTimestamp = (value: string): string => {
  if (!/^\d{4}-\d{2}-\d{2} /.test(value)) {
    return value;
  }

  const date = new Date(`${value.replace(' ', 'T')}Z`);

  return Number.isNaN(date.getTime()) ? value : date.toISOString();
};

const toSummary = (dto: ProjectSummaryDTO): ProjectSummary => ({
  createdAt: normalizeServerTimestamp(dto.created_at),
  id: dto.project_id,
  name: dto.name,
  revision: dto.revision,
  updatedAt: normalizeServerTimestamp(dto.updated_at),
});

/** Most recently edited first — the order Home and the Open dialog present. */
const sortSummaries = (summaries: ProjectSummary[]): ProjectSummary[] =>
  [...summaries].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

export const useProjectLibrary = (): ProjectLibrarySnapshot => store.useSnapshot();

export const useProjectLibrarySelector = store.useSelector;

export const getProjectLibrary = (): ProjectLibrarySnapshot => store.getSnapshot();

/** Adopt summaries already fetched elsewhere (the workbench boot pass). */
export const seedProjectLibrary = (dtos: ProjectSummaryDTO[]): void => {
  store.setSnapshot({ status: 'ready', summaries: sortSummaries(dtos.map(toSummary)) });
};

const refreshFlight = createSingleFlight<void>();

/** Re-list from the server; concurrent calls share one request. */
export const refreshProjectLibrary = (): Promise<void> =>
  refreshFlight.run('project-library', () =>
    listProjects()
      .then((dtos) => {
        seedProjectLibrary(dtos);
      })
      .catch((error: unknown) => {
        store.patchSnapshot({
          error: error instanceof Error ? error.message : 'Failed to load projects.',
          status: 'error',
        });
      })
  );

/**
 * Reflect a save the editor just pushed, so the library stays current without
 * a refetch. `updatedAt` is stamped locally; the next refresh replaces it
 * with the server's value.
 */
export const upsertProjectSummary = (entry: { id: string; name: string; revision: number | null }): void => {
  const { summaries } = store.getSnapshot();
  const existing = summaries.find((summary) => summary.id === entry.id);
  const updatedAt = new Date().toISOString();
  const next: ProjectSummary = {
    createdAt: existing?.createdAt ?? updatedAt,
    id: entry.id,
    name: entry.name,
    revision: entry.revision ?? existing?.revision ?? 0,
    updatedAt,
  };

  store.patchSnapshot({
    status: 'ready',
    summaries: sortSummaries([...summaries.filter((summary) => summary.id !== entry.id), next]),
  });
};

/** Permanently remove a project from the server. The only deletion path. */
export const deleteLibraryProject = async (projectId: string): Promise<void> => {
  await apiDeleteProject(projectId);
  store.patchSnapshot({ summaries: store.getSnapshot().summaries.filter((summary) => summary.id !== projectId) });
};

/**
 * Rename a project that is not open in the editor. Open projects rename
 * through the workbench reducer instead, so their document and the autosave
 * revision chain stay consistent.
 */
export const renameLibraryProject = async (projectId: string, name: string): Promise<void> => {
  const record = await apiGetProject(projectId);
  const updated = await apiUpdateProject(projectId, {
    data: { ...record.data, name },
    expected_revision: record.revision,
    name,
  });

  upsertProjectSummary({ id: updated.project_id, name: updated.name, revision: updated.revision });
};

/** Copy a project under a fresh id; returns the new summary. */
export const duplicateLibraryProject = async (projectId: string): Promise<ProjectSummary> => {
  const record = await apiGetProject(projectId);
  const newId = createProjectId();
  const name = `${record.name} copy`;
  const created = await apiCreateProject({
    data: { ...record.data, id: newId, name },
    name,
    project_id: newId,
  });
  const summary = toSummary(created);

  upsertProjectSummary({ id: summary.id, name: summary.name, revision: summary.revision });

  return summary;
};
