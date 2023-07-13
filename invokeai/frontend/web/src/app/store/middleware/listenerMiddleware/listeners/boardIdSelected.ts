import { log } from 'app/logging/useLogger';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  boardIdSelected,
  imageSelected,
  selectImagesAll,
} from 'features/gallery/store/gallerySlice';
import { boardsApi } from 'services/api/endpoints/boards';
import {
  IMAGES_PER_PAGE,
  receivedPageOfImages,
} from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const board_id = action.payload;

      // we need to check if we need to fetch more images

      const state = getState();
      const allImages = selectImagesAll(state);

      if (board_id === 'all') {
        // Selected all images
        dispatch(imageSelected(allImages[0]?.image_name ?? null));
        return;
      }

      if (board_id === 'batch') {
        // Selected the batch
        dispatch(imageSelected(state.gallery.batchImageNames[0] ?? null));
        return;
      }

      const filteredImages = selectFilteredImages(state);

      const categories =
        state.gallery.galleryView === 'images'
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

      // get the board from the cache
      const { data: boards } =
        boardsApi.endpoints.listAllBoards.select()(state);
      const board = boards?.find((b) => b.board_id === board_id);

      if (!board) {
        // can't find the board in cache...
        dispatch(boardIdSelected('all'));
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
