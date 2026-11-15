import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  createBoard,
  GALLERY_DEMO_BOARDS,
  GALLERY_DEMO_ITEMS,
  GALLERY_DEMO_PROJECT,
  getBoardCounts,
  getBoardGroups,
  getGridEmptyState,
  getGridSections,
  toggleStarred,
} from './gallery-demo.mjs';

const PROJECT = GALLERY_DEMO_PROJECT.id;
const ids = (/** @type {{ id: string }[]} */ list) => list.map((entry) => entry.id);
const filter = (overrides = {}) => ({
  boardId: 'concept-art',
  view: /** @type {const} */ ('images'),
  starredOnly: false,
  searchTerm: '',
  order: /** @type {const} */ ('DESC'),
  ...overrides,
});

describe('getBoardGroups', () => {
  it('puts the project inbox first and Uncategorized first in the Library, whatever the list order', () => {
    const shuffled = [...GALLERY_DEMO_BOARDS].reverse();
    const groups = getBoardGroups(shuffled, PROJECT, '');
    assert.deepEqual(ids(groups.projectBoards), ['inbox', 'landscapes', 'portraits', 'concept-art']);
    assert.deepEqual(ids(groups.libraryBoards), ['uncategorized', 'product-shots', 'architecture']);
  });

  it('counts boards per tier, the inbox included and Uncategorized not', () => {
    const groups = getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, '');
    assert.equal(groups.projectCount, 4);
    assert.equal(groups.libraryCount, 2);
  });

  it('filters by name, ignoring case and surrounding space, and recounts', () => {
    const groups = getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, '  PORT ');
    assert.deepEqual(ids(groups.projectBoards), ['portraits']);
    assert.deepEqual(ids(groups.libraryBoards), []);
    assert.equal(groups.projectCount, 1);
    assert.equal(groups.libraryCount, 0);
    assert.equal(groups.hasAnyMatch, true);
  });

  it('offers to create a board unless a shown board already has that exact name', () => {
    assert.equal(getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, '').canCreateFromSearch, false);
    assert.equal(getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, 'portraits').canCreateFromSearch, false);
    assert.equal(getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, 'portrait').canCreateFromSearch, true);
    const none = getBoardGroups(GALLERY_DEMO_BOARDS, PROJECT, 'Storyboards');
    assert.equal(none.hasAnyMatch, false);
    assert.equal(none.canCreateFromSearch, true);
  });
});

describe('createBoard', () => {
  it('adds an empty board to the chosen tier that the groups then list', () => {
    const { boards, board } = createBoard(GALLERY_DEMO_BOARDS, '  Storyboards ', null);
    assert.equal(board.name, 'Storyboards');
    assert.equal(boards.length, GALLERY_DEMO_BOARDS.length + 1);
    assert.deepEqual(getBoardCounts(GALLERY_DEMO_ITEMS, board.id), { media: 0, uploads: 0 });
    const groups = getBoardGroups(boards, PROJECT, 'story');
    assert.deepEqual(ids(groups.libraryBoards), [board.id]);
    assert.equal(groups.canCreateFromSearch, true);

    const second = createBoard(boards, 'Moodboards', PROJECT);
    assert.notEqual(second.board.id, board.id);
    assert.ok(ids(getBoardGroups(second.boards, PROJECT, '').projectBoards).includes(second.board.id));
  });
});

describe('getBoardCounts', () => {
  it('splits a board into media and uploads', () => {
    assert.deepEqual(getBoardCounts(GALLERY_DEMO_ITEMS, 'concept-art'), { media: 5, uploads: 1 });
    assert.deepEqual(getBoardCounts(GALLERY_DEMO_ITEMS, 'uncategorized'), { media: 1, uploads: 2 });
    assert.deepEqual(getBoardCounts(GALLERY_DEMO_ITEMS, 'nowhere'), { media: 0, uploads: 0 });
  });
});

describe('getGridSections', () => {
  it('lists the board and view newest first, and pins its starred items above', () => {
    const { starred, listing } = getGridSections(GALLERY_DEMO_ITEMS, filter());
    assert.deepEqual(ids(listing), ['c1', 'c2', 'c3', 'c4', 'c5']);
    assert.deepEqual(ids(starred), ['c2', 'c3']);
  });

  it('reverses for oldest first and switches to uploads', () => {
    assert.deepEqual(ids(getGridSections(GALLERY_DEMO_ITEMS, filter({ order: 'ASC' })).listing), [
      'c5',
      'c4',
      'c3',
      'c2',
      'c1',
    ]);
    assert.deepEqual(ids(getGridSections(GALLERY_DEMO_ITEMS, filter({ view: 'assets' })).listing), ['c6']);
  });

  it('shows only starred items, without a strip, under the starred filter', () => {
    assert.deepEqual(getGridSections(GALLERY_DEMO_ITEMS, filter({ starredOnly: true })), {
      starred: [],
      listing: getGridSections(GALLERY_DEMO_ITEMS, filter()).starred,
    });
  });

  it('searches names and prompts, applying to the strip too', () => {
    const { starred, listing } = getGridSections(GALLERY_DEMO_ITEMS, filter({ searchTerm: 'Torch' }));
    assert.deepEqual(ids(listing), ['c3']);
    assert.deepEqual(ids(starred), ['c3']);
    assert.deepEqual(ids(getGridSections(GALLERY_DEMO_ITEMS, filter({ searchTerm: 'robot' })).listing), ['c2']);
  });

  it('follows a toggled star', () => {
    const items = toggleStarred(toggleStarred(GALLERY_DEMO_ITEMS, 'c2'), 'c1');
    assert.deepEqual(ids(getGridSections(items, filter()).starred), ['c1', 'c3']);
    assert.equal(GALLERY_DEMO_ITEMS.find((entry) => entry.id === 'c1')?.starred, false, 'the input is not mutated');
  });
});

describe('getGridEmptyState', () => {
  it('invites an upload only when nothing filters the board', () => {
    assert.equal(getGridEmptyState(3, { starredOnly: true, searchTerm: 'x' }), null);
    assert.equal(getGridEmptyState(0, { starredOnly: false, searchTerm: '' })?.kind, 'upload');
    assert.match(getGridEmptyState(0, { starredOnly: true, searchTerm: '' })?.text ?? '', /No starred items/);
    assert.match(getGridEmptyState(0, { starredOnly: false, searchTerm: ' car ' })?.text ?? '', /No items match/);
  });
});
