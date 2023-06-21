import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { boardIdSelected } from 'features/gallery/store/boardSlice';
import { selectImagesAll } from 'features/gallery/store/imagesSlice';
import { receivedPageOfImages } from 'services/thunks/image';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const boardId = action.payload;
      const state = getState();
      const { categories } = state.images;

      const images = selectImagesAll(state).filter((i) => {
        const isInCategory = categories.includes(i.image_category);
        const isInSelectedBoard = boardId ? i.board_id === boardId : true;
        return isInCategory && isInSelectedBoard;
      });

      if (images.length === 0) {
        dispatch(receivedPageOfImages({ categories, boardId }));
      }
    },
  });
};
