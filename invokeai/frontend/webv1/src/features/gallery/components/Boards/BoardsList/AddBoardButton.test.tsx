import { autoAddBoardIdChanged, boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createBoardAndDispatchActions, getCreatedBoardActions } from './AddBoardButton';

describe('getCreatedBoardActions', () => {
  it('selects the created board and auto-adds it when auto assign is enabled', () => {
    expect(getCreatedBoardActions('board-1', true)).toEqual([
      boardIdSelected({ boardId: 'board-1' }),
      autoAddBoardIdChanged('board-1'),
      boardSearchTextChanged(''),
    ]);
  });

  it('does not change the auto-add board when auto assign is disabled', () => {
    expect(getCreatedBoardActions('board-1', false)).toEqual([
      boardIdSelected({ boardId: 'board-1' }),
      boardSearchTextChanged(''),
    ]);
  });
});

describe('createBoardAndDispatchActions', () => {
  const createBoard = vi.fn();
  const dispatch = vi.fn();

  beforeEach(() => {
    createBoard.mockReset();
    dispatch.mockReset();
  });

  it('auto-adds the created board when auto assign is enabled', async () => {
    createBoard.mockReturnValue({
      unwrap: () => Promise.resolve({ board_id: 'board-1' }),
    });

    await createBoardAndDispatchActions(createBoard, dispatch, 'My Board', true);

    expect(createBoard).toHaveBeenCalledWith({ board_name: 'My Board' });
    expect(dispatch.mock.calls).toEqual([
      [boardIdSelected({ boardId: 'board-1' })],
      [autoAddBoardIdChanged('board-1')],
      [boardSearchTextChanged('')],
    ]);
  });

  it('leaves the auto-add board unchanged when auto assign is disabled', async () => {
    createBoard.mockReturnValue({
      unwrap: () => Promise.resolve({ board_id: 'board-1' }),
    });

    await createBoardAndDispatchActions(createBoard, dispatch, 'My Board', false);

    expect(dispatch.mock.calls).toEqual([[boardIdSelected({ boardId: 'board-1' })], [boardSearchTextChanged('')]]);
  });

  it('swallows errors from createBoard', async () => {
    createBoard.mockReturnValue({
      unwrap: () => Promise.reject(new Error('boom')),
    });

    await expect(createBoardAndDispatchActions(createBoard, dispatch, 'My Board', true)).resolves.toBeUndefined();
    expect(dispatch).not.toHaveBeenCalled();
  });
});
