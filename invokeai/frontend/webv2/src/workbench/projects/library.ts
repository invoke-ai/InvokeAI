import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';

import {
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  getProject as apiGetProject,
  listProjects,
  updateProject as apiUpdateProject,
  type ProjectRecordDTO,
  type ProjectSummaryDTO,
} from './api';
import {
  forgetProjectCover,
  getProjectCoverImageName,
  getProjectCoverUrl,
  loadProjectCovers,
  subscribeProjectCovers,
} from './covers';
import { createProjectId } from './ids';
import { refreshOpenProjects } from './openProjects';
import { pruneSessionProject } from './session';
import { getOpenProject } from './syncStore';

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
  /**
   * Thumbnail for the library grid, resolved from the per-user cover index
   * (`covers.ts`) rather than from the project document, which listings do not
   * carry. Absent for a project that has produced no images, and for one last
   * saved by a build that did not record covers.
   */
  coverUrl?: string;
}

export type ProjectLibraryStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface ProjectLibrarySnapshot {
  status: ProjectLibraryStatus;
  summaries: ProjectSummary[];
  error?: string;
}

const EMPTY_PROJECT_LIBRARY: ProjectLibrarySnapshot = { status: 'idle', summaries: [] };
const store = createExternalStore<ProjectLibrarySnapshot>(EMPTY_PROJECT_LIBRARY);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_PROJECT_LIBRARY);
  },
  name: 'project-library',
});

const withCover = <T extends { id: string }>(summary: T): T & { coverUrl?: string } => {
  const coverImageName = getProjectCoverImageName(summary.id);

  return coverImageName === undefined ? summary : { ...summary, coverUrl: getProjectCoverUrl(coverImageName) };
};

const toSummary = (dto: ProjectSummaryDTO): ProjectSummary =>
  withCover({
    createdAt: normalizeServerTimestamp(dto.created_at),
    id: dto.project_id,
    name: dto.name,
    revision: dto.revision,
    updatedAt: normalizeServerTimestamp(dto.updated_at),
  });

/**
 * The cover index loads and updates independently of the listing, so summaries
 * are re-derived whenever it changes. Without this a cover recorded after the
 * library rendered — the first generation in a new project — would not appear
 * until the next refresh.
 */
subscribeProjectCovers(() => {
  const { status, summaries } = store.getSnapshot();

  if (status !== 'ready') {
    return;
  }

  const next = summaries.map((summary) => {
    const coverImageName = getProjectCoverImageName(summary.id);
    const coverUrl = coverImageName === undefined ? undefined : getProjectCoverUrl(coverImageName);

    return coverUrl === summary.coverUrl ? summary : { ...summary, coverUrl };
  });

  if (next.some((summary, index) => summary !== summaries[index])) {
    store.patchSnapshot({ summaries: next });
  }
});

/** Most recently edited first — the order Home and the Open dialog present. */
const sortSummaries = (summaries: ProjectSummary[]): ProjectSummary[] =>
  [...summaries].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

export const useProjectLibrary = (): ProjectLibrarySnapshot => store.useSnapshot();

export const useProjectLibrarySelector = store.useSelector;

export const getProjectLibrary = (): ProjectLibrarySnapshot => store.getSnapshot();

/** Adopt summaries already fetched elsewhere (the workbench boot pass). */
export const seedProjectLibrary = (dtos: ProjectSummaryDTO[], owner: AccountScope): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  store.setSnapshot({ status: 'ready', summaries: sortSummaries(dtos.map(toSummary)) });
};

const refreshFlight = createSingleFlight<void>();

/** Re-list from the server; concurrent calls share one request. */
export const refreshProjectLibrary = (): Promise<void> =>
  (() => {
    const owner = captureAccountScope();

    return refreshFlight.run(`project-library:${owner.epoch}`, () =>
      // Covers resolve alongside the listing rather than after it: they come
      // from a different store, and seeding without them would show a grid of
      // glyphs that fills in a moment later.
      Promise.all([listProjects(owner.signal), loadProjectCovers()])
        .then(([dtos]) => {
          seedProjectLibrary(dtos, owner);
        })
        .catch((error: unknown) => {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          store.patchSnapshot({
            error: error instanceof Error ? error.message : 'Failed to load projects.',
            status: 'error',
          });
        })
    );
  })();

/**
 * Reflect a save the editor just pushed, so the library stays current without
 * a refetch. `updatedAt` is stamped locally; the next refresh replaces it
 * with the server's value.
 */
export const upsertProjectSummary = (
  entry: { id: string; name: string; revision: number | null },
  owner: AccountScope
): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const { summaries } = store.getSnapshot();
  const existing = summaries.find((summary) => summary.id === entry.id);
  const updatedAt = new Date().toISOString();
  const next: ProjectSummary = withCover({
    createdAt: existing?.createdAt ?? updatedAt,
    id: entry.id,
    name: entry.name,
    revision: entry.revision ?? existing?.revision ?? 0,
    updatedAt,
  });

  store.patchSnapshot({
    status: 'ready',
    summaries: sortSummaries([...summaries.filter((summary) => summary.id !== entry.id), next]),
  });
};

/**
 * Every mutation below branches on one question: does the workbench currently hold this project?
 *
 * If it does, the mutation goes through {@link getOpenProject} — the sync engine — and if it does
 * not, it goes over HTTP. A library write landing beside an open project's revision chain used to
 * fork it into a conflict copy; now that a project's board renames with it, such a write would
 * rename the board too, from outside the transaction that is supposed to own both.
 */

/** Permanently remove a project from the server, its board with it. The only deletion path. */
export const deleteLibraryProject = async (projectId: string): Promise<void> => {
  const owner = captureAccountScope();
  const openProject = getOpenProject(projectId);

  // Marked before the request so an autosave in flight cannot recreate the project — and with it a
  // board — between the DELETE and the tab closing.
  openProject?.markDeleted();
  await apiDeleteProject(projectId, owner.signal);
  assertAccountScopeCurrent(owner);
  openProject?.close();
  forgetProjectCover(projectId, owner);
  store.patchSnapshot({ summaries: store.getSnapshot().summaries.filter((summary) => summary.id !== projectId) });

  // The saved session outlives the editor, so a deleted project has to leave it here or the next
  // boot tries to open something the server no longer has.
  await pruneSessionProject(projectId, owner.signal);
  await refreshOpenProjects();
};

/**
 * Rename a project. An open one renames through the reducer so its document, its revision chain and
 * its board stay in step; a closed one is a read-modify-write against the server.
 */
export const renameLibraryProject = async (projectId: string, name: string): Promise<void> => {
  const owner = captureAccountScope();
  const openProject = getOpenProject(projectId);

  if (openProject) {
    await openProject.rename(name);
    assertAccountScopeCurrent(owner);
    // The flush has already told the server; this only keeps the grid from waiting for a refetch.
    upsertProjectSummary({ id: projectId, name, revision: null }, owner);

    return;
  }

  const record = await apiGetProject(projectId, owner.signal);

  assertAccountScopeCurrent(owner);
  const updated = await apiUpdateProject(
    projectId,
    {
      data: { ...record.data, name },
      expected_revision: record.revision,
      name,
    },
    owner.signal
  );

  assertAccountScopeCurrent(owner);
  upsertProjectSummary({ id: updated.project_id, name: updated.name, revision: updated.revision }, owner);
};

/**
 * The acknowledged record for a project about to be copied.
 *
 * An open project is flushed first and the flush is fatal if it fails. Duplicating what the server
 * last acknowledged would silently drop everything the person can see on screen, and a copy that
 * quietly omits the last ten minutes of work is worse than no copy.
 */
export const readProjectForDuplication = async (projectId: string, owner: AccountScope): Promise<ProjectRecordDTO> => {
  await getOpenProject(projectId)?.flush();
  assertAccountScopeCurrent(owner);

  return apiGetProject(projectId, owner.signal);
};

/** Adopt a project this account just created elsewhere (an import, a duplication). */
export const adoptCreatedProject = (record: ProjectRecordDTO, owner: AccountScope): ProjectSummary => {
  const summary = toSummary(record);

  upsertProjectSummary({ id: summary.id, name: summary.name, revision: summary.revision }, owner);

  return summary;
};

/** Copy a project under a fresh id; returns the new summary. */
export const duplicateLibraryProject = async (projectId: string): Promise<ProjectSummary> => {
  const owner = captureAccountScope();
  const record = await readProjectForDuplication(projectId, owner);

  assertAccountScopeCurrent(owner);
  const newId = createProjectId();
  const name = `${record.name} copy`;
  const created = await apiCreateProject(
    {
      data: { ...record.data, id: newId, name },
      name,
      project_id: newId,
    },
    owner.signal
  );

  assertAccountScopeCurrent(owner);

  return adoptCreatedProject(created, owner);
};
