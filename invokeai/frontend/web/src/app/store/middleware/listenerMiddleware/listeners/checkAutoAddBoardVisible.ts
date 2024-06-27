import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { checkAutoAddBoardVisible } from 'features/gallery/store/actions';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { boardsApi } from 'services/api/endpoints/boards';

export const addCheckAutoAddBoardVisibleListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: checkAutoAddBoardVisible,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();
      const queryArgs = selectListBoardsQueryArgs(state);
      const queryResult = boardsApi.endpoints.listAllBoards.select(queryArgs)(state);
      const autoAddBoardId = state.gallery.autoAddBoardId;

      if (!queryResult.data) {
        return;
      }

      if (!queryResult.data.find((board) => board.board_id === autoAddBoardId)) {
        dispatch(autoAddBoardIdChanged('none'));
      }
    },
  });
};
