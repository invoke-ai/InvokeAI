import { skipToken } from '@reduxjs/toolkit/dist/query';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
} from 'features/gallery/store/gallerySlice';
import { useMemo } from 'react';
import { ListImagesArgs, useListImagesQuery } from '../endpoints/images';

const baseQueryArg: ListImagesArgs = {
  offset: 0,
  limit: INITIAL_IMAGE_LIMIT,
  is_intermediate: false,
};

const imagesQueryArg: ListImagesArgs = {
  categories: IMAGE_CATEGORIES,
  ...baseQueryArg,
};

const assetsQueryArg: ListImagesArgs = {
  categories: ASSETS_CATEGORIES,
  ...baseQueryArg,
};

const noBoardQueryArg: ListImagesArgs = {
  board_id: 'none',
  ...baseQueryArg,
};

export const useBoardTotal = (board_id: BoardId | null | undefined) => {
  const queryArg = useMemo(() => {
    if (!board_id) {
      return;
    }
    if (board_id === 'images') {
      return imagesQueryArg;
    } else if (board_id === 'assets') {
      return assetsQueryArg;
    } else if (board_id === 'no_board') {
      return noBoardQueryArg;
    } else {
      return { board_id, ...baseQueryArg };
    }
  }, [board_id]);

  const { total } = useListImagesQuery(queryArg ?? skipToken, {
    selectFromResult: ({ currentData }) => ({ total: currentData?.total }),
  });

  return total;
};
