import { log } from 'app/logging/useLogger';
import {
  boardIdSelected,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import {
  getBoardIdQueryParamForBoard,
  getCategoriesQueryParamForBoard,
} from 'features/gallery/store/util';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const _board_id = action.payload;
      const state = getState();
      const categories = getCategoriesQueryParamForBoard(_board_id);
      const board_id = getBoardIdQueryParamForBoard(_board_id);
      const queryArgs = { board_id, categories };

      const { data: boardImagesData } =
        imagesApi.endpoints.listImages.select(queryArgs)(state);

      if (boardImagesData?.ids.length) {
        dispatch(imageSelected((boardImagesData.ids[0] as string) ?? null));
      } else {
        dispatch(imageSelected(null));
      }
    },
  });
};
