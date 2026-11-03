import type { ProjectLibrarySnapshot } from '@workbench/projects/library';
import type { TFunction } from 'i18next';

import { isTimestampInRange } from '@platform/search/dateTokens';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { getProjectLibrary, isProjectSummaryCompatible, refreshProjectLibrary } from '@workbench/projects/library';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import { createTermsMatcher, PaletteSearchUnavailableError } from './entries';
import { getObjectIdentity } from './objectIdentity';

/** A library that is not ready is fetched before searching, so the section stays in its searching state. */
const loadReadyLibrary = async (
  library: ProjectLibrarySnapshot,
  signal: AbortSignal,
  t: TFunction
): Promise<ProjectLibrarySnapshot> => {
  if (library.status === 'ready') {
    return library;
  }

  // Single-flight; Retry runs the search again, and with it the refresh.
  await refreshProjectLibrary();
  signal.throwIfAborted();
  const current = getProjectLibrary();

  if (current.status !== 'ready') {
    throw new PaletteSearchUnavailableError(t('commandPalette.states.projectsUnavailable'));
  }

  return current;
};

/**
 * Library summaries plus the workbench's open projects, which the library may not list yet. The key follows the
 * summaries rather than the whole snapshot: a library that loads re-runs the search, a repeated failure does not.
 * The editor and Launchpad palettes both use it, so this module imports no editor runtime.
 */
export const createProjectsProvider = ({
  library,
  openProject,
  openProjects,
  t,
}: {
  library: ProjectLibrarySnapshot;
  openProject: (projectId: string, name: string) => void;
  openProjects: readonly { id: string; name: string }[];
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: `${getObjectIdentity(library.summaries, 'library')}:${openProjects.map(({ id, name }) => `${id}\u0000${name}`).join('\u0001')}`,
  label: t('commandPalette.groups.projects'),
  providerKey: getPaletteContributionKey('provider', 'projects'),
  supportsCreatedAtRange: true,
  search: async (query, { signal }) => {
    signal.throwIfAborted();
    const { summaries } = await loadReadyLibrary(library, signal, t);
    const matchesQuery = createTermsMatcher(query.text);
    const openNames = new Map(openProjects.map(({ id, name }) => [id, name] as const));
    const librarySummaryIds = new Set(summaries.map((summary) => summary.id));
    // Open projects the library does not list yet lead; the library follows in its last-edited order.
    const candidates = [
      ...openProjects
        .filter(({ id }) => !librarySummaryIds.has(id))
        .map(({ id, name }) => ({ createdAt: undefined, id, name, updatedAt: undefined })),
      // The Launchpad offers no way into a project this build cannot open; neither does the palette.
      ...summaries.filter(isProjectSummaryCompatible).map(({ createdAt, id, name, updatedAt }) => ({
        createdAt,
        id,
        // An open project's name is current before the library refetches it.
        name: openNames.get(id) ?? name,
        updatedAt,
      })),
    ];

    return candidates
      .filter(
        ({ createdAt, name }) =>
          matchesQuery(name) &&
          // Open projects missing from the library have no known creation date, so a date range excludes them.
          (query.range === undefined || (createdAt !== undefined && isTimestampInRange(createdAt, query.range)))
      )
      .map<PaletteEntry>(({ id, name, updatedAt }) => ({
        group: 'Projects',
        groupLabel: t('commandPalette.groups.projects'),
        id: `project:${id}`,
        isPersistentRecent: false,
        run: () => openProject(id, name),
        subtitle:
          openNames.has(id) || updatedAt === undefined
            ? t('commandPalette.providers.projectOpen')
            : t('projects.editedRelative', { time: formatRelativeTime(updatedAt) }),
        title: name,
      }));
  },
});
