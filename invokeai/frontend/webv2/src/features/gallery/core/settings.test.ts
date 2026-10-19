import { describe, expect, it } from 'vitest';

import { getGallerySettings, isGalleryBoardSectionOpen, toggleGalleryBoardSection } from './settings';

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
