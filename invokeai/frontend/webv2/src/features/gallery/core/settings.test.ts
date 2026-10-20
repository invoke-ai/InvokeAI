import { describe, expect, it } from 'vitest';

import {
  getGallerySettings,
  isGalleryBoardSectionOpen,
  toggleGalleryBoardSection,
  toggleGalleryProjectGroup,
} from './settings';

describe('board panel sections', () => {
  it('starts other users’ boards collapsed, even for a panel that saved its collapsed groups before', () => {
    const settings = getGallerySettings({ collapsedBoardSections: ['dates'] });

    expect(isGalleryBoardSectionOpen(settings, 'other-users')).toBe(false);
    expect(isGalleryBoardSectionOpen(settings, 'shared')).toBe(true);
    expect(isGalleryBoardSectionOpen(settings, 'dates')).toBe(false);
  });

  it('records the groups that start collapsed as opened, and the rest as collapsed', () => {
    const settings = getGallerySettings({});

    const opened = getGallerySettings({ ...settings, ...toggleGalleryBoardSection(settings, 'other-users', true) });
    const closed = getGallerySettings({ ...opened, ...toggleGalleryBoardSection(opened, 'shared', false) });

    expect(isGalleryBoardSectionOpen(opened, 'other-users')).toBe(true);
    expect(isGalleryBoardSectionOpen(closed, 'shared')).toBe(false);
    expect(closed).toMatchObject({ collapsedBoardSections: ['shared'], expandedBoardSections: ['other-users'] });
  });
});

describe('project groups under Other projects', () => {
  it('starts with every group expanded', () => {
    expect(getGallerySettings({}).collapsedProjectGroupIds).toEqual([]);
  });

  it('keeps distinct project ids only, dropping anything that is not a string', () => {
    const settings = getGallerySettings({ collapsedProjectGroupIds: ['a', 7, null, 'b', 'a', { id: 'c' }] });

    expect(settings.collapsedProjectGroupIds).toEqual(['b', 'a']);
    expect(getGallerySettings({ collapsedProjectGroupIds: 'a' }).collapsedProjectGroupIds).toEqual([]);
  });

  it('remembers at most 200 groups, keeping the most recently collapsed', () => {
    const ids = Array.from({ length: 205 }, (_, index) => `project-${String(index)}`);

    const stored = getGallerySettings({ collapsedProjectGroupIds: ids }).collapsedProjectGroupIds;
    const collapsedOneMore = toggleGalleryProjectGroup(
      getGallerySettings({ collapsedProjectGroupIds: ids }),
      'new',
      false
    );

    expect(stored).toHaveLength(200);
    expect(stored[0]).toBe('project-5');
    expect(collapsedOneMore.collapsedProjectGroupIds).toHaveLength(200);
    expect(collapsedOneMore.collapsedProjectGroupIds.at(-1)).toBe('new');
    expect(collapsedOneMore.collapsedProjectGroupIds).not.toContain('project-5');
  });

  it('collapses and expands one group without touching the others', () => {
    const settings = getGallerySettings({ collapsedProjectGroupIds: ['a'] });

    expect(toggleGalleryProjectGroup(settings, 'b', false).collapsedProjectGroupIds).toEqual(['a', 'b']);
    expect(toggleGalleryProjectGroup(settings, 'a', true).collapsedProjectGroupIds).toEqual([]);
  });
});

describe('search scope', () => {
  it('searches the selected board unless all boards were chosen', () => {
    expect(getGallerySettings({}).searchScope).toBe('board');
    expect(getGallerySettings({ searchScope: 'all' }).searchScope).toBe('all');
    expect(getGallerySettings({ searchScope: 'everywhere' }).searchScope).toBe('board');
    expect(getGallerySettings({ searchScope: null }).searchScope).toBe('board');
  });
});
