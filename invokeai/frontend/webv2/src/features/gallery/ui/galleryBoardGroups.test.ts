import type { GalleryBoard } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import { getGalleryBoardGroups } from './galleryBoardGroups';

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  projectId: null,
  videoCount: 0,
  ...overrides,
});

const uncategorized = createBoard({ id: 'none', kind: 'uncategorized', name: '' });
const dogs = createBoard({ id: 'dogs', name: 'dogs' });
const cats = createBoard({ id: 'cats', name: 'Cats' });
const archived = createBoard({ archived: true, id: 'gorl', name: 'GORL' });
const dateBoard = createBoard({ id: 'by_date:2026-07-28', kind: 'date', name: '28 July' });
const mine = createBoard({ id: 'mine', isInbox: true, name: 'My project', projectId: 'project-1' });
const mineMember = createBoard({ id: 'mine-member', name: 'Façades', projectId: 'project-1' });
const mineArchived = createBoard({ archived: true, id: 'mine-old', name: 'Old façades', projectId: 'project-1' });
const zebra = createBoard({ id: 'zebra', isInbox: true, name: 'Zebra study', projectId: 'project-z' });
const zebraMember = createBoard({ id: 'zebra-member', name: 'Stripes', projectId: 'project-z' });
const apple = createBoard({ id: 'apple', isInbox: true, name: 'Apple ads', projectId: 'project-a' });
const t = (key: string) =>
  key === 'widgets.gallery.uncategorized' ? 'Uncategorized' : key === 'widgets.gallery.inbox' ? 'Inbox' : key;

const ALL = [uncategorized, dogs, cats, archived, dateBoard, zebraMember, mineMember, zebra, mine, apple, mineArchived];

const groupsOf = (overrides: Partial<Parameters<typeof getGalleryBoardGroups>[0]> = {}) =>
  getGalleryBoardGroups({
    boards: ALL,
    projectBoardId: 'mine',
    projectId: 'project-1',
    searchTerm: '',
    showArchived: true,
    showDates: true,
    showOtherProjects: true,
    t,
    ...overrides,
  });

const ids = (boards: GalleryBoard[]) => boards.map((board) => board.id);

describe('getGalleryBoardGroups', () => {
  it('puts the open project inbox first, then its members, and Uncategorized first in the Library', () => {
    const groups = groupsOf();

    expect(ids(groups.projectBoards)).toEqual(['mine', 'mine-member']);
    expect(ids(groups.libraryBoards)).toEqual(['none', 'dogs', 'cats']);
  });

  it('claims the open project inbox by id before the listing knows its membership', () => {
    const draftInbox = createBoard({ id: 'draft', isInbox: true, name: 'Draft' });
    const groups = groupsOf({
      boards: [uncategorized, draftInbox, dogs],
      projectBoardId: 'draft',
      projectId: 'p-draft',
    });

    expect(ids(groups.projectBoards)).toEqual(['draft']);
    expect(ids(groups.libraryBoards)).toEqual(['none', 'dogs']);
  });

  it('groups other projects by name with each inbox first, only when they are shown', () => {
    const groups = groupsOf();

    expect(groups.otherProjects.map((group) => [group.projectId, ids(group.boards)])).toEqual([
      ['project-a', ['apple']],
      ['project-z', ['zebra', 'zebra-member']],
    ]);
    expect(groupsOf({ showOtherProjects: false }).otherProjects).toEqual([]);
  });

  it('keeps other projects archived boards out of the archived section while they are hidden', () => {
    const otherArchived = createBoard({ archived: true, id: 'their-old', name: 'Theirs', projectId: 'project-z' });

    expect(ids(groupsOf({ boards: [...ALL, otherArchived] }).archivedBoards)).toEqual([
      'gorl',
      'mine-old',
      'their-old',
    ]);
    expect(ids(groupsOf({ boards: [...ALL, otherArchived], showOtherProjects: false }).archivedBoards)).toEqual([
      'gorl',
      'mine-old',
    ]);
  });

  it('treats every project as other where no project is open', () => {
    const groups = groupsOf({ projectBoardId: null, projectId: null });

    expect(groups.projectBoards).toEqual([]);
    expect(groups.otherProjects.map((group) => group.projectId)).toEqual(['project-a', 'project-1', 'project-z']);
  });

  it('hides the date and archived sections when their toggles are off', () => {
    const groups = groupsOf({ showArchived: false, showDates: false });

    expect(groups.archivedBoards).toEqual([]);
    expect(groups.dateBoards).toEqual([]);
  });

  it('filters every section by a case-insensitive substring match on the shown label', () => {
    const groups = groupsOf({ searchTerm: 'DOG' });

    expect(ids(groups.libraryBoards)).toEqual(['dogs']);
    expect(groups.projectBoards).toEqual([]);
    expect(groups.otherProjects).toEqual([]);
    expect(groups.hasAnyMatch).toBe(true);

    // Inboxes answer to "Inbox", not to the project name the server stores them under.
    expect(ids(groupsOf({ searchTerm: 'inbox' }).projectBoards)).toEqual(['mine']);
    expect(groupsOf({ searchTerm: 'inbox' }).otherProjects.map((group) => ids(group.boards))).toEqual([
      ['apple'],
      ['zebra'],
    ]);
    expect(groupsOf({ searchTerm: 'zebra' }).hasAnyMatch).toBe(false);
  });

  it('matches anywhere in the name, so Uncategorized answers to "cat"', () => {
    expect(ids(groupsOf({ searchTerm: 'cat' }).libraryBoards)).toEqual(['none', 'cats']);
  });

  it('offers to create only when the search names no existing board', () => {
    expect(groupsOf({ searchTerm: 'birds' }).canCreateFromSearch).toBe(true);
    expect(groupsOf({ searchTerm: 'cats' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: '  CATS  ' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: 'Uncategorized' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: 'Inbox' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: '' }).canCreateFromSearch).toBe(false);
  });

  it('lets a name that exists only in a hidden tier be created', () => {
    // "Stripes" lives in project-z, hidden while other projects are off; "GORL" is archived and hidden.
    expect(groupsOf({ searchTerm: 'Stripes', showOtherProjects: false }).canCreateFromSearch).toBe(true);
    expect(groupsOf({ searchTerm: 'GORL', showArchived: false }).canCreateFromSearch).toBe(true);
    expect(groupsOf({ searchTerm: 'Stripes' }).canCreateFromSearch).toBe(false);
  });

  it('reports no match when the search excludes every row', () => {
    const groups = groupsOf({ searchTerm: 'nothing-here' });

    expect(groups.hasAnyMatch).toBe(false);
    expect(groups.canCreateFromSearch).toBe(true);
  });
});
