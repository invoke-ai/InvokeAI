import type { ProjectSummary } from '@workbench/projects/library';

import {
  buildProjectGroups,
  flattenProjectGroupsToRows,
  isProjectSortId,
  isProjectsViewId,
  matchesProjectSearch,
  prunePinnedProjectIds,
  toggleProjectPin,
  type ProjectLibraryViewInput,
} from '@workbench/launchpad/projects/projectLibraryView';
import { describe, expect, it } from 'vitest';

const NOW = Date.parse('2026-08-04T15:00:00.000Z');

const at = (isoOffsetDays: number, hour = 12): string =>
  new Date(
    new Date(NOW).setHours(0, 0, 0, 0) - isoOffsetDays * 24 * 60 * 60 * 1000 + hour * 60 * 60 * 1000
  ).toISOString();

const summary = (overrides: Partial<ProjectSummary> & { id: string }): ProjectSummary => ({
  createdAt: at(0),
  name: overrides.id,
  revision: 1,
  updatedAt: at(0),
  ...overrides,
});

const view = (overrides: Partial<ProjectLibraryViewInput> = {}): ProjectLibraryViewInput => ({
  now: NOW,
  openProjectIds: [],
  pinnedIds: [],
  searchTerm: '',
  sort: 'edited',
  summaries: [],
  ...overrides,
});

describe('matchesProjectSearch', () => {
  it('matches case-insensitively on a substring and ignores surrounding whitespace', () => {
    const project = summary({ id: 'a', name: 'Sunset Cabin' });

    expect(matchesProjectSearch(project, 'cabin')).toBe(true);
    expect(matchesProjectSearch(project, '  SUNSET ')).toBe(true);
    expect(matchesProjectSearch(project, 'lighthouse')).toBe(false);
  });

  it('treats an empty search as matching everything', () => {
    expect(matchesProjectSearch(summary({ id: 'a' }), '   ')).toBe(true);
  });
});

describe('buildProjectGroups', () => {
  it('buckets by calendar recency rather than a rolling 24 hours', () => {
    const groups = buildProjectGroups(
      view({
        summaries: [
          summary({ id: 'today', updatedAt: at(0, 1) }),
          summary({ id: 'week', updatedAt: at(3) }),
          summary({ id: 'month', updatedAt: at(20) }),
          summary({ id: 'older', updatedAt: at(400) }),
        ],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['today', 'week', 'month', 'older']);
  });

  it('puts pinned projects first and takes them out of their date bucket', () => {
    const groups = buildProjectGroups(
      view({
        pinnedIds: ['b'],
        summaries: [summary({ id: 'a', updatedAt: at(0) }), summary({ id: 'b', updatedAt: at(0) })],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['pinned', 'today']);
    expect(groups[0]?.projects.map((project) => project.id)).toEqual(['b']);
    expect(groups[1]?.projects.map((project) => project.id)).toEqual(['a']);
  });

  it('lists a project that is both pinned and open only under pinned', () => {
    const groups = buildProjectGroups(
      view({ openProjectIds: ['a'], pinnedIds: ['a'], summaries: [summary({ id: 'a' })] })
    );

    expect(groups).toHaveLength(1);
    expect(groups[0]?.id).toBe('pinned');
  });

  it('separates open projects from the date buckets', () => {
    const groups = buildProjectGroups(
      view({
        openProjectIds: ['open'],
        summaries: [summary({ id: 'open', updatedAt: at(20) }), summary({ id: 'other', updatedAt: at(20) })],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['open', 'month']);
  });

  it('collapses date buckets into one group while searching', () => {
    const groups = buildProjectGroups(
      view({
        searchTerm: 'cabin',
        summaries: [
          summary({ id: 'a', name: 'Cabin today', updatedAt: at(0) }),
          summary({ id: 'b', name: 'Cabin last year', updatedAt: at(400) }),
          summary({ id: 'c', name: 'Lighthouse', updatedAt: at(0) }),
        ],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['all']);
    expect(groups[0]?.projects.map((project) => project.id)).toEqual(['a', 'b']);
  });

  it('collapses date buckets when sorting by name, and orders naturally', () => {
    const groups = buildProjectGroups(
      view({
        sort: 'name',
        summaries: [
          summary({ id: 'c', name: 'Project 10' }),
          summary({ id: 'a', name: 'project 2' }),
          summary({ id: 'b', name: 'Project 1' }),
        ],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['all']);
    expect(groups[0]?.projects.map((project) => project.name)).toEqual(['Project 1', 'project 2', 'Project 10']);
  });

  it('buckets and orders on createdAt when sorting by created', () => {
    const groups = buildProjectGroups(
      view({
        sort: 'created',
        summaries: [
          summary({ createdAt: at(400), id: 'old-but-edited-today', updatedAt: at(0) }),
          summary({ createdAt: at(0), id: 'new', updatedAt: at(400) }),
        ],
      })
    );

    expect(groups.map((group) => group.id)).toEqual(['today', 'older']);
    expect(groups[0]?.projects.map((project) => project.id)).toEqual(['new']);
  });

  it('drops empty groups instead of emitting bare headings', () => {
    const groups = buildProjectGroups(view({ summaries: [summary({ id: 'a', updatedAt: at(400) })] }));

    expect(groups.map((group) => group.id)).toEqual(['older']);
  });

  it('returns nothing when the search matches nothing', () => {
    expect(buildProjectGroups(view({ searchTerm: 'zzz', summaries: [summary({ id: 'a', name: 'Cabin' })] }))).toEqual(
      []
    );
  });
});

describe('flattenProjectGroupsToRows', () => {
  it('chunks each group into rows of the given column count, under a heading', () => {
    const groups = buildProjectGroups(
      view({
        pinnedIds: ['a'],
        summaries: [
          summary({ id: 'a' }),
          summary({ id: 'b' }),
          summary({ id: 'c' }),
          summary({ id: 'd' }),
          summary({ id: 'e' }),
        ],
      })
    );

    expect(flattenProjectGroupsToRows(groups, 2)).toEqual([
      { count: 1, group: 'pinned', kind: 'header' },
      { group: 'pinned', kind: 'projects', projects: [expect.objectContaining({ id: 'a' })] },
      { count: 4, group: 'today', kind: 'header' },
      {
        group: 'today',
        kind: 'projects',
        projects: [expect.objectContaining({ id: 'b' }), expect.objectContaining({ id: 'c' })],
      },
      {
        group: 'today',
        kind: 'projects',
        projects: [expect.objectContaining({ id: 'd' }), expect.objectContaining({ id: 'e' })],
      },
    ]);
  });

  it('omits the heading for a lone unnamed group', () => {
    const groups = buildProjectGroups(view({ searchTerm: 'a', summaries: [summary({ id: 'a', name: 'Cabin' })] }));

    expect(flattenProjectGroupsToRows(groups, 3)).toEqual([
      { group: 'all', kind: 'projects', projects: [expect.objectContaining({ id: 'a' })] },
    ]);
  });

  it('treats a nonsensical column count as one column rather than looping forever', () => {
    const groups = buildProjectGroups(view({ summaries: [summary({ id: 'a' }), summary({ id: 'b' })] }));

    expect(flattenProjectGroupsToRows(groups, 0).filter((row) => row.kind === 'projects')).toHaveLength(2);
  });
});

describe('toggleProjectPin', () => {
  it('appends a new pin and removes an existing one', () => {
    expect(toggleProjectPin(['a'], 'b')).toEqual(['a', 'b']);
    expect(toggleProjectPin(['a', 'b'], 'a')).toEqual(['b']);
  });

  it('collapses a duplicated id to a single entry', () => {
    expect(toggleProjectPin(['a', 'a'], 'a')).toEqual([]);
  });
});

describe('prunePinnedProjectIds', () => {
  it('drops pins whose project is gone and keeps the rest in order', () => {
    expect(prunePinnedProjectIds(['a', 'gone', 'b'], [summary({ id: 'b' }), summary({ id: 'a' })])).toEqual(['a', 'b']);
  });
});

describe('identifier guards', () => {
  it('accepts known ids and rejects anything else', () => {
    expect(isProjectSortId('edited')).toBe(true);
    expect(isProjectSortId('sideways')).toBe(false);
    expect(isProjectsViewId('list')).toBe(true);
    expect(isProjectsViewId(null)).toBe(false);
  });
});
