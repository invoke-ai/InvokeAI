import { autoAddBoardIdChanged, boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  autoAssignBoardOnClick: true,
  createBoard: vi.fn(),
  dispatch: vi.fn(),
}));

vi.mock('react', async () => {
  const actual = await vi.importActual('react');
  return {
    ...actual,
    memo: (component: unknown) => component,
    useCallback: (callback: unknown) => callback,
  };
});

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => (key === 'boards.myBoard' ? 'My Board' : key),
  }),
}));

vi.mock('app/store/storeHooks', () => ({
  useAppDispatch: () => mocks.dispatch,
  useAppSelector: () => mocks.autoAssignBoardOnClick,
}));

vi.mock('services/api/endpoints/boards', () => ({
  useCreateBoardMutation: () => [mocks.createBoard, { isLoading: false }],
}));

import AddBoardButton, { getCreatedBoardActions } from './AddBoardButton';

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

describe('AddBoardButton', () => {
  beforeEach(() => {
    mocks.autoAssignBoardOnClick = true;
    mocks.createBoard.mockReset();
    mocks.dispatch.mockReset();
  });

  it('auto-adds the created board when auto assign is enabled', async () => {
    mocks.createBoard.mockReturnValue({
      unwrap: () => Promise.resolve({ board_id: 'board-1' }),
    });

    const button = AddBoardButton({}) as { props: { onClick: () => Promise<void> } };
    await button.props.onClick();

    expect(mocks.createBoard).toHaveBeenCalledWith({ board_name: 'My Board' });
    expect(mocks.dispatch.mock.calls).toEqual([
      [boardIdSelected({ boardId: 'board-1' })],
      [autoAddBoardIdChanged('board-1')],
      [boardSearchTextChanged('')],
    ]);
  });

  it('leaves the auto-add board unchanged when auto assign is disabled', async () => {
    mocks.autoAssignBoardOnClick = false;
    mocks.createBoard.mockReturnValue({
      unwrap: () => Promise.resolve({ board_id: 'board-1' }),
    });

    const button = AddBoardButton({}) as { props: { onClick: () => Promise<void> } };
    await button.props.onClick();

    expect(mocks.dispatch.mock.calls).toEqual([
      [boardIdSelected({ boardId: 'board-1' })],
      [boardSearchTextChanged('')],
    ]);
  });
});
