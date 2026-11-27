/**
 * Pins the translation of a selected board into name-list query args.
 *
 * A virtual board is a date, not a board row. Regular boards and virtual dates now share one
 * endpoint (`listGalleryItemNames`), so the only thing keeping virtual dates working is that
 * the id is converted into a `created_date` filter and *not* forwarded as `board_id` — the
 * backend would filter on a board that does not exist and return an empty gallery.
 *
 * The original bug this area guards (PR #9163 review): virtual boards were image-only, so
 * videos created on that date never appeared. The server-side half of that guarantee is pinned
 * by tests/app/routers/test_virtual_boards.py.
 */
import { createStore } from 'app/store/store';
import { selectGalleryItemNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { describe, expect, it } from 'vitest';

describe('selectGalleryItemNamesQueryArgs', () => {
  it('converts a virtual board id into a created_date filter', () => {
    const store = createStore();
    store.dispatch(boardIdSelected({ boardId: 'by_date:2026-07-26' }));

    const args = selectGalleryItemNamesQueryArgs(store.getState());

    expect(args.created_date).toBe('2026-07-26');
    expect(args.board_id).toBeUndefined();
  });

  it('passes a regular board id through untouched', () => {
    const store = createStore();
    store.dispatch(boardIdSelected({ boardId: 'some-board-uuid' }));

    const args = selectGalleryItemNamesQueryArgs(store.getState());

    expect(args.board_id).toBe('some-board-uuid');
    expect(args.created_date).toBeUndefined();
  });
});
