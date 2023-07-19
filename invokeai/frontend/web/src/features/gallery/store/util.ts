import { SYSTEM_BOARDS } from 'services/api/endpoints/images';
import { ASSETS_CATEGORIES, BoardId, IMAGE_CATEGORIES } from './gallerySlice';

export const getCategoriesQueryParamForBoard = (board_id: BoardId) => {
  if (board_id === 'assets') {
    return ASSETS_CATEGORIES;
  }

  if (board_id === 'images') {
    return IMAGE_CATEGORIES;
  }

  // 'no_board' board, 'batch' board, user boards
  return undefined;
};

export const getBoardIdQueryParamForBoard = (board_id: BoardId) => {
  if (board_id === 'no_board') {
    return 'none';
  }

  // system boards besides 'no_board'
  if (SYSTEM_BOARDS.includes(board_id)) {
    return undefined;
  }

  // user boards
  return board_id;
};
