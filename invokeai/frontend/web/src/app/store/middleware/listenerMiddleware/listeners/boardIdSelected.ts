import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { boardIdSelected } from 'features/gallery/store/boardSlice';
import { selectImagesAll } from 'features/gallery/store/imagesSlice';
import { IMAGES_PER_PAGE, receivedPageOfImages } from 'services/thunks/image';
import { api } from 'services/apiSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const boardId = action.payload;

      // we need to check if we need to fetch more images

      const state = getState();
      const allImages = selectImagesAll(state);

      if (!boardId) {
        // a board was unselected
        dispatch(imageSelected(allImages[0]?.image_name));
        return;
      }

      const { categories } = state.images;

      const filteredImages = allImages.filter((i) => {
        const isInCategory = categories.includes(i.image_category);
        const isInSelectedBoard = boardId ? i.board_id === boardId : true;
        return isInCategory && isInSelectedBoard;
      });

      // get the board from the cache
      const { data: boards } = api.endpoints.listAllBoards.select()(state);
      const board = boards?.find((b) => b.board_id === boardId);

      if (!board) {
        // can't find the board in cache...
        dispatch(imageSelected(allImages[0]?.image_name));
        return;
      }

      dispatch(imageSelected(board.cover_image_name));

      // if we haven't loaded one full page of images from this board, load more
      if (
        filteredImages.length < board.image_count &&
        filteredImages.length < IMAGES_PER_PAGE
      ) {
        dispatch(receivedPageOfImages({ categories, boardId }));
      }
    },
  });
};

export const addBoardIdSelected_changeSelectedImage_listener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const boardId = action.payload;

      const state = getState();

      // we need to check if we need to fetch more images

      if (!boardId) {
        // a board was unselected - we don't need to do anything
        return;
      }

      const { categories } = state.images;

      const filteredImages = selectImagesAll(state).filter((i) => {
        const isInCategory = categories.includes(i.image_category);
        const isInSelectedBoard = boardId ? i.board_id === boardId : true;
        return isInCategory && isInSelectedBoard;
      });

      // get the board from the cache
      const { data: boards } = api.endpoints.listAllBoards.select()(state);
      const board = boards?.find((b) => b.board_id === boardId);
      if (!board) {
        // can't find the board in cache...
        return;
      }

      // if we haven't loaded one full page of images from this board, load more
      if (
        filteredImages.length < board.image_count &&
        filteredImages.length < IMAGES_PER_PAGE
      ) {
        dispatch(receivedPageOfImages({ categories, boardId }));
      }
    },
  });
};
