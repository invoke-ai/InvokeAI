import { describe, expect, it } from 'vitest';

import { getProjectSwitcherSections } from './projectSwitcherModel';

describe('project switcher sections', () => {
  it('lists every open project and excludes them from recents', () => {
    const openProjects = [
      { id: 'open-1', name: 'First' },
      { id: 'open-2', name: 'Second' },
    ];
    const libraryProjects = [
      { id: 'open-2', name: 'Second', updatedAt: 2 },
      { id: 'saved-1', name: 'Saved', updatedAt: 1 },
    ];

    expect(getProjectSwitcherSections(openProjects, libraryProjects, 5)).toEqual({
      open: openProjects,
      recent: [{ id: 'saved-1', name: 'Saved', updatedAt: 1 }],
    });
  });

  it('limits only the recent library section', () => {
    const openProjects = Array.from({ length: 7 }, (_, index) => ({ id: `open-${index}`, name: `${index}` }));
    const libraryProjects = Array.from({ length: 7 }, (_, index) => ({
      id: `saved-${index}`,
      name: `${index}`,
      updatedAt: index,
    }));

    const sections = getProjectSwitcherSections(openProjects, libraryProjects, 5);

    expect(sections.open).toHaveLength(7);
    expect(sections.recent).toHaveLength(5);
  });
});
