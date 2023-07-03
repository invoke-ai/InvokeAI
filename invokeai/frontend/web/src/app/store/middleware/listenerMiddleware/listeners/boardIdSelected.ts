import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import {
  imageSelected,
  selectImagesAll,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import {
  IMAGES_PER_PAGE,
  receivedPageOfImages,
} from 'services/api/thunks/image';
import { boardsApi } from 'services/api/endpoints/boards';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const board_id = action.payload;

      // we need to check if we need to fetch more images

      const state = getState();
      const allImages = selectImagesAll(state);

      if (!board_id) {
        // a board was unselected
        dispatch(imageSelected(allImages[0]?.image_name));
        return;
      }

      const { categories } = state.gallery;

      const filteredImages = allImages.filter((i) => {
        const isInCategory = categories.includes(i.image_category);
        const isInSelectedBoard = board_id ? i.board_id === board_id : true;
        return isInCategory && isInSelectedBoard;
      });

      // get the board from the cache
      const { data: boards } =
        boardsApi.endpoints.listAllBoards.select()(state);
      const board = boards?.find((b) => b.board_id === board_id);

      if (!board) {
        // can't find the board in cache...
        dispatch(imageSelected(allImages[0]?.image_name));
        return;
      }

      dispatch(imageSelected(board.cover_image_name ?? null));

      // if we haven't loaded one full page of images from this board, load more
      if (
        filteredImages.length < board.image_count &&
        filteredImages.length < IMAGES_PER_PAGE
      ) {
        dispatch(
          receivedPageOfImages({ categories, board_id, is_intermediate: false })
        );
      }
    },
  });
};

export const addBoardIdSelected_changeSelectedImage_listener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const board_id = action.payload;

      const state = getState();

      // we need to check if we need to fetch more images

      if (!board_id) {
        // a board was unselected - we don't need to do anything
        return;
      }

      const { categories } = state.gallery;

      const filteredImages = selectImagesAll(state).filter((i) => {
        const isInCategory = categories.includes(i.image_category);
        const isInSelectedBoard = board_id ? i.board_id === board_id : true;
        return isInCategory && isInSelectedBoard;
      });

      // get the board from the cache
      const { data: boards } =
        boardsApi.endpoints.listAllBoards.select()(state);
      const board = boards?.find((b) => b.board_id === board_id);
      if (!board) {
        // can't find the board in cache...
        return;
      }

      // if we haven't loaded one full page of images from this board, load more
      if (
        filteredImages.length < board.image_count &&
        filteredImages.length < IMAGES_PER_PAGE
      ) {
        dispatch(
          receivedPageOfImages({ categories, board_id, is_intermediate: false })
        );
      }
    },
  });
};
