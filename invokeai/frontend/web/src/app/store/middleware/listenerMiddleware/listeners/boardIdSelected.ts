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
    effect: async (
      action,
      { getState, dispatch, condition, cancelActiveListeners }
    ) => {
      // Cancel any in-progress instances of this listener, we don't want to select an image from a previous board
      cancelActiveListeners();

      const _board_id = action.payload;
      // when a board is selected, we need to wait until the board has loaded *some* images, then select the first one

      const categories = getCategoriesQueryParamForBoard(_board_id);
      const board_id = getBoardIdQueryParamForBoard(_board_id);
      const queryArgs = { board_id, categories };

      // wait until the board has some images - maybe it already has some from a previous fetch
      // must use getState() to ensure we do not have stale state
      const isSuccess = await condition(
        () =>
          imagesApi.endpoints.listImages.select(queryArgs)(getState())
            .isSuccess,
        1000
      );

      if (isSuccess) {
        // the board was just changed - we can select the first image
        const { data: boardImagesData } = imagesApi.endpoints.listImages.select(
          queryArgs
        )(getState());

        if (boardImagesData?.ids.length) {
          dispatch(imageSelected((boardImagesData.ids[0] as string) ?? null));
        } else {
          // board has no images - deselect
          dispatch(imageSelected(null));
        }
      } else {
        // fallback - deselect
        dispatch(imageSelected(null));
      }
    },
  });
};
