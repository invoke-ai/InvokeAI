import { describe, expect, it } from 'vitest';

import type { GalleryBoard, GalleryBoardAction } from './types';

import { getGalleryBoardDestinationGroups, getGalleryBoardLabel, getGalleryProjectGroupLabel } from './boardLabels';

const createBoard = (overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  assetVideoCount: 0,
  id: 'dogs',
  imageCount: 50,
  kind: 'board',
  name: 'dogs',
  isInbox: false,
  projectId: null,
  videoCount: 4,
  ...overrides,
});

const t = (key: string) =>
  key === 'widgets.gallery.uncategorized'
    ? 'Uncategorized'
    : key === 'widgets.gallery.inbox'
      ? 'Inbox'
      : key === 'widgets.gallery.boardGroups.library'
        ? 'Library'
        : key === 'widgets.gallery.unknownProject'
          ? 'Another project'
          : key;

describe('getGalleryBoardLabel', () => {
  it('uses the stored name for real boards', () => {
    expect(getGalleryBoardLabel(createBoard(), t)).toBe('dogs');
  });

  it('translates the synthesized uncategorized board instead of trusting its name', () => {
    expect(getGalleryBoardLabel(createBoard({ kind: 'uncategorized', name: '' }), t)).toBe('Uncategorized');
  });
});

describe('getGalleryBoardLabel for inboxes', () => {
  it('calls every inbox Inbox, whatever name the server stores it under', () => {
    expect(getGalleryBoardLabel(createBoard({ isInbox: true, name: 'Mahogany House', projectId: 'p1' }), t)).toBe(
      'Inbox'
    );
  });
});

describe('getGalleryBoardLabel for someone else’s inbox', () => {
  it('keeps its stored name, the project’s, which nothing else on screen names', () => {
    const access = {
      actions: ['read' as const],
      adminOverride: true,
      generalAccess: 'restricted' as const,
      role: null,
    };

    expect(getGalleryBoardLabel(createBoard({ access, isInbox: true, name: 'Their project' }), t)).toBe(
      'Their project'
    );
  });
});

describe('getGalleryProjectGroupLabel', () => {
  const inbox = createBoard({ id: 'inbox', isInbox: true, name: 'Stored name', projectId: 'p2' });

  it('prefers the account project list, which follows a rename at once', () => {
    expect(getGalleryProjectGroupLabel('p2', [inbox], new Map([['p2', 'Renamed']]), t)).toBe('Renamed');
  });

  it('falls back to the inbox stored name, then to a stand-in', () => {
    expect(getGalleryProjectGroupLabel('p2', [inbox], new Map(), t)).toBe('Stored name');
    expect(getGalleryProjectGroupLabel('p2', [createBoard({ projectId: 'p2' })], new Map(), t)).toBe('Another project');
  });
});

describe('getGalleryBoardDestinationGroups', () => {
  const boards = [
    createBoard({ id: 'zebra-member', name: 'Stripes', projectId: 'pz' }),
    createBoard({ id: 'mine-member', name: 'Façades', projectId: 'p1' }),
    createBoard({ id: 'cats', name: 'Cats' }),
    createBoard({ id: 'zebra', isInbox: true, name: 'Zebra', projectId: 'pz' }),
    createBoard({ id: 'mine', isInbox: true, name: 'Mine', projectId: 'p1' }),
    createBoard({ id: 'apple-member', name: 'Pips', projectId: 'pa' }),
  ];
  const groupsOf = (projectId: string | null, projectNames = new Map<string, string>()) =>
    getGalleryBoardDestinationGroups({ boards, projectId, projectName: 'Mine', projectNames, t }).map((group) => [
      group.label,
      group.boards.map((board) => board.id),
    ]);

  it('lists the open project with its inbox first, then the Library, then other projects by name', () => {
    expect(groupsOf('p1', new Map([['pa', 'Apple']]))).toEqual([
      ['Mine', ['mine', 'mine-member']],
      ['Library', ['cats']],
      ['Apple', ['apple-member']],
      ['Zebra', ['zebra', 'zebra-member']],
    ]);
  });

  it('offers only boards the caller may add to, other people’s apart from their own', () => {
    const access = (role: 'viewer' | 'contributor' | null, actions: GalleryBoardAction[]) => ({
      access: { actions, adminOverride: role === null, generalAccess: 'viewer' as const, role },
    });
    const withShared = [
      ...boards,
      createBoard({ id: 'moodboard', name: 'Moodboard', ...access('contributor', ['read', 'add_media']) }),
      createBoard({ id: 'references', name: 'References', ...access('viewer', ['read']) }),
      createBoard({ id: 'theirs', name: 'Theirs', projectId: 'their-project', ...access(null, ['read', 'add_media']) }),
    ];

    expect(
      getGalleryBoardDestinationGroups({
        boards: withShared,
        projectId: 'p1',
        projectName: 'Mine',
        projectNames: new Map(),
        t,
      }).map((group) => [group.id, group.boards.map((board) => board.id)])
    ).toEqual([
      ['project', ['mine', 'mine-member']],
      ['library', ['cats']],
      ['shared', ['moodboard']],
      ['other-users', ['theirs']],
      ['project:pa', ['apple-member']],
      ['project:pz', ['zebra', 'zebra-member']],
    ]);
  });

  it('drops empty tiers and treats every project as other where none is open', () => {
    expect(groupsOf(null)).toEqual([
      ['Library', ['cats']],
      ['Another project', ['apple-member']],
      ['Mine', ['mine', 'mine-member']],
      ['Zebra', ['zebra', 'zebra-member']],
    ]);
    expect(
      getGalleryBoardDestinationGroups({ boards: [], projectId: 'p1', projectName: 'Mine', projectNames: new Map(), t })
    ).toEqual([]);
  });
});
