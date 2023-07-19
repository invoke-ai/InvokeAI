import { SYSTEM_BOARDS } from 'services/api/endpoints/images';
import { ASSETS_CATEGORIES, BoardId, IMAGE_CATEGORIES } from './gallerySlice';
import { ImageCategory } from 'services/api/types';
import { isEqual } from 'lodash-es';

export const getCategoriesQueryParamForBoard = (
  board_id: BoardId
): ImageCategory[] | undefined => {
  if (board_id === 'assets') {
    return ASSETS_CATEGORIES;
  }

  if (board_id === 'images') {
    return IMAGE_CATEGORIES;
  }

  // 'no_board' board, 'batch' board, user boards
  return undefined;
};

export const getBoardIdQueryParamForBoard = (
  board_id: BoardId
): string | undefined => {
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

export const getBoardIdFromBoardAndCategoriesQueryParam = (
  board_id: string | undefined,
  categories: ImageCategory[] | undefined
): BoardId => {
  if (board_id === undefined && isEqual(categories, IMAGE_CATEGORIES)) {
    return 'images';
  }

  if (board_id === undefined && isEqual(categories, ASSETS_CATEGORIES)) {
    return 'assets';
  }

  if (board_id === 'none') {
    return 'no_board';
  }

  return board_id ?? 'UNKNOWN_BOARD';
};
