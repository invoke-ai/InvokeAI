import type { ProjectSummary } from '@workbench/projects/library';

/**
 * How the project library is presented: what matches the search, what order it
 * is in, and which bucket each project falls into.
 *
 * All of it is pure and takes `now` as an argument rather than reading the
 * clock, so the date bucketing is testable without freezing time.
 */

export type ProjectSortId = 'edited' | 'created' | 'name';

export type ProjectsViewId = 'grid' | 'list';

export const PROJECT_SORT_IDS: readonly ProjectSortId[] = ['edited', 'created', 'name'];

export const PROJECTS_VIEW_IDS: readonly ProjectsViewId[] = ['grid', 'list'];

export const isProjectSortId = (value: unknown): value is ProjectSortId =>
  typeof value === 'string' && PROJECT_SORT_IDS.includes(value as ProjectSortId);

export const isProjectsViewId = (value: unknown): value is ProjectsViewId =>
  typeof value === 'string' && PROJECTS_VIEW_IDS.includes(value as ProjectsViewId);

/**
 * `pinned` and `open` are cross-cutting: a project lands in one of them
 * instead of a date bucket, never in both. The rest are relative to `now`.
 * `all` is the flat bucket used when a search or an alphabetical sort makes
 * date headings meaningless.
 */
export type ProjectGroupId = 'pinned' | 'open' | 'today' | 'week' | 'month' | 'older' | 'all';

export interface ProjectGroup {
  id: ProjectGroupId;
  projects: ProjectSummary[];
}

export interface ProjectLibraryViewInput {
  summaries: readonly ProjectSummary[];
  searchTerm: string;
  sort: ProjectSortId;
  pinnedIds: readonly string[];
  openProjectIds: readonly string[];
  now: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;

const getSortTimestamp = (summary: ProjectSummary, sort: ProjectSortId): number => {
  const value = Date.parse(sort === 'created' ? summary.createdAt : summary.updatedAt);

  return Number.isNaN(value) ? 0 : value;
};

const compareProjects = (a: ProjectSummary, b: ProjectSummary, sort: ProjectSortId): number => {
  if (sort === 'name') {
    return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
  }

  return getSortTimestamp(b, sort) - getSortTimestamp(a, sort);
};

/** Local midnight, so "today" means the calendar day rather than 24 hours. */
const startOfLocalDay = (timestamp: number): number => {
  const date = new Date(timestamp);

  date.setHours(0, 0, 0, 0);

  return date.getTime();
};

const getDateGroupId = (summary: ProjectSummary, sort: ProjectSortId, now: number): ProjectGroupId => {
  const timestamp = getSortTimestamp(summary, sort);
  const todayStart = startOfLocalDay(now);

  if (timestamp >= todayStart) {
    return 'today';
  }

  if (timestamp >= todayStart - 6 * DAY_MS) {
    return 'week';
  }

  if (timestamp >= todayStart - 29 * DAY_MS) {
    return 'month';
  }

  return 'older';
};

export const matchesProjectSearch = (summary: ProjectSummary, searchTerm: string): boolean =>
  summary.name.toLowerCase().includes(searchTerm.trim().toLowerCase());

const DATE_GROUP_ORDER: readonly ProjectGroupId[] = ['today', 'week', 'month', 'older'];

/**
 * Group order is fixed: pinned, then what is already open, then recency. A
 * search or an alphabetical sort collapses the date buckets into one `all`
 * group — ranking twelve results under four date headings hides them.
 */
export const buildProjectGroups = ({
  now,
  openProjectIds,
  pinnedIds,
  searchTerm,
  sort,
  summaries,
}: ProjectLibraryViewInput): ProjectGroup[] => {
  const matched = summaries.filter((summary) => matchesProjectSearch(summary, searchTerm));
  const pinned = new Set(pinnedIds);
  const open = new Set(openProjectIds);
  const sorted = [...matched].sort((a, b) => compareProjects(a, b, sort));
  const isSearching = searchTerm.trim().length > 0;
  const buckets = new Map<ProjectGroupId, ProjectSummary[]>();
  const push = (id: ProjectGroupId, summary: ProjectSummary): void => {
    const bucket = buckets.get(id);

    if (bucket) {
      bucket.push(summary);
    } else {
      buckets.set(id, [summary]);
    }
  };

  for (const summary of sorted) {
    if (pinned.has(summary.id)) {
      push('pinned', summary);
      continue;
    }

    if (open.has(summary.id)) {
      push('open', summary);
      continue;
    }

    push(isSearching || sort === 'name' ? 'all' : getDateGroupId(summary, sort, now), summary);
  }

  const order: readonly ProjectGroupId[] = ['pinned', 'open', 'all', ...DATE_GROUP_ORDER];

  return order.flatMap((id) => {
    const projects = buckets.get(id);

    return projects ? [{ id, projects }] : [];
  });
};

export type ProjectRowModel =
  | { kind: 'header'; group: ProjectGroupId; count: number }
  | { kind: 'projects'; group: ProjectGroupId; projects: ProjectSummary[] };

/**
 * Flatten groups into virtualizer rows. `columnCount` is how many cards share
 * a row, so the grid virtualizes by row and the list is just the same shape
 * with one column.
 *
 * A single unnamed group (the search/alphabetical `all` case) renders without
 * a heading — there is nothing to distinguish it from.
 */
export const flattenProjectGroupsToRows = (groups: readonly ProjectGroup[], columnCount: number): ProjectRowModel[] => {
  const perRow = Math.max(1, columnCount);
  const showHeadings = groups.length > 1 || groups[0]?.id !== 'all';

  return groups.flatMap((group) => {
    const rows: ProjectRowModel[] = showHeadings
      ? [{ count: group.projects.length, group: group.id, kind: 'header' }]
      : [];

    for (let index = 0; index < group.projects.length; index += perRow) {
      rows.push({ group: group.id, kind: 'projects', projects: group.projects.slice(index, index + perRow) });
    }

    return rows;
  });
};

/** Toggle membership while preserving insertion order, so pins stay stable. */
export const toggleProjectPin = (pinnedIds: readonly string[], projectId: string): string[] =>
  pinnedIds.includes(projectId)
    ? pinnedIds.filter((id) => id !== projectId)
    : [...pinnedIds.filter((id) => id !== projectId), projectId];

/** Drop pins for projects that no longer exist, so the list cannot grow forever. */
export const prunePinnedProjectIds = (pinnedIds: readonly string[], summaries: readonly ProjectSummary[]): string[] => {
  const existing = new Set(summaries.map((summary) => summary.id));

  return pinnedIds.filter((id) => existing.has(id));
};
