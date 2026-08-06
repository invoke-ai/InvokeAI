import type { GalleryBoard } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import { getGalleryBoardGroups } from './galleryBoardGroups';

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  imageCount: 0,
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
const t = (key: string) => (key === 'widgets.gallery.uncategorized' ? 'Uncategorized' : key);

const groupsOf = (overrides: Partial<Parameters<typeof getGalleryBoardGroups>[0]> = {}) =>
  getGalleryBoardGroups({
    boards: [uncategorized, dogs, cats, archived, dateBoard],
    projectBoardId: null,
    projectName: 'Project',
    searchTerm: '',
    showArchived: true,
    showDates: true,
    t,
    ...overrides,
  });

describe('getGalleryBoardGroups', () => {
  it('splits archived boards into their own section instead of the main list', () => {
    const groups = groupsOf();

    expect(groups.yourBoards.map((board) => board.id)).toEqual(['dogs', 'cats', 'none']);
    expect(groups.archivedBoards.map((board) => board.id)).toEqual(['gorl']);
  });

  it('puts Uncategorized last and hoists the project board first', () => {
    const groups = groupsOf({ projectBoardId: 'cats' });

    expect(groups.yourBoards.map((board) => board.id)).toEqual(['cats', 'dogs', 'none']);
  });

  it('hides the date and archived sections when their toggles are off', () => {
    const groups = groupsOf({ showArchived: false, showDates: false });

    expect(groups.archivedBoards).toEqual([]);
    expect(groups.dateBoards).toEqual([]);
    expect(groups.yourBoards.map((board) => board.id)).toEqual(['dogs', 'cats', 'none']);
  });

  it('filters every section by a case-insensitive substring match', () => {
    const groups = groupsOf({ searchTerm: 'DOG' });

    expect(groups.yourBoards.map((board) => board.id)).toEqual(['dogs']);
    expect(groups.archivedBoards).toEqual([]);
    expect(groups.dateBoards).toEqual([]);
    expect(groups.hasAnyMatch).toBe(true);
  });

  it('matches anywhere in the name, so Uncategorized answers to "cat"', () => {
    expect(groupsOf({ searchTerm: 'cat' }).yourBoards.map((board) => board.id)).toEqual(['cats', 'none']);
  });

  it('offers to create only when the search names no existing board', () => {
    expect(groupsOf({ searchTerm: 'birds' }).canCreateFromSearch).toBe(true);
    expect(groupsOf({ searchTerm: 'cats' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: '  CATS  ' }).canCreateFromSearch).toBe(false);
    expect(groupsOf({ searchTerm: '' }).canCreateFromSearch).toBe(false);
  });

  it('treats the localized Uncategorized label as an exact match', () => {
    expect(groupsOf({ searchTerm: 'Uncategorized' }).canCreateFromSearch).toBe(false);
  });

  it('treats the yet-to-exist project board as an exact match so it is never duplicated', () => {
    const groups = groupsOf({ searchTerm: 'Project' });

    expect(groups.hasProjectBoardPlaceholder).toBe(true);
    expect(groups.canCreateFromSearch).toBe(false);
    expect(groups.hasAnyMatch).toBe(true);
  });

  it('drops the project placeholder once the project board exists', () => {
    expect(groupsOf({ projectBoardId: 'cats' }).hasProjectBoardPlaceholder).toBe(false);
  });

  it('reports no match when the search excludes every row', () => {
    const groups = groupsOf({ searchTerm: 'nothing-here' });

    expect(groups.hasAnyMatch).toBe(false);
    expect(groups.canCreateFromSearch).toBe(true);
  });
});
