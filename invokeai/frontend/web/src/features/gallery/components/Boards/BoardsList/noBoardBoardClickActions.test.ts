import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { describe, expect, it } from 'vitest';

import { getNoBoardClickActions } from './noBoardBoardClickActions';

describe('getNoBoardClickActions', () => {
  it('selects the board when it is not already selected', () => {
    expect(getNoBoardClickActions(false, false)).toEqual([boardIdSelected({ boardId: 'none' })]);
  });

  it('does not re-select the board the user is already on', () => {
    // GalleryBoard and VirtualBoardItem have always guarded this; NoBoardBoard did not, so
    // clicking Uncategorized while already on it restarted the gallery's auto-select probe. That
    // probe selects the board's first item unconditionally, so it discarded whatever the user had
    // selected — and moving the displayed item mid-generation also lifts the progress overlay for
    // a couple of seconds (PR #9520 review).
    expect(getNoBoardClickActions(true, false)).toEqual([]);
  });

  it('still assigns the auto-add board when the board is already selected', () => {
    // The guard is only on the selection half: clicking the current board must still be able to
    // make it the auto-add target, which is the other thing this click means.
    expect(getNoBoardClickActions(true, true)).toEqual([autoAddBoardIdChanged('none')]);
  });

  it('selects and auto-adds when both apply', () => {
    expect(getNoBoardClickActions(false, true)).toEqual([
      boardIdSelected({ boardId: 'none' }),
      autoAddBoardIdChanged('none'),
    ]);
  });
});
