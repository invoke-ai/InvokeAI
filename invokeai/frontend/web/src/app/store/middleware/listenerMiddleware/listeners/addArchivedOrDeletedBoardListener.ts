import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
  galleryViewChanged,
  shouldShowArchivedBoardsChanged,
} from 'features/gallery/store/gallerySlice';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';

export const addArchivedOrDeletedBoardListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(
      // Updating a board may change its archived status
      boardsApi.endpoints.updateBoard.matchFulfilled,
      // If the selected/auto-add board was deleted from a different session, we'll only know during the list request,
      boardsApi.endpoints.listAllBoards.matchFulfilled,
      // If a board is deleted, we'll need to reset the auto-add board
      imagesApi.endpoints.deleteBoard.matchFulfilled,
      imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
      // When we change the visibility of archived boards, we may need to reset the auto-add board
      shouldShowArchivedBoardsChanged
    ),
    effect: async (action, { dispatch, getState }) => {
      /**
       * The auto-add board shouldn't be set to an archived board or deleted board. When we archive a board, delete
       * a board, or change a the archived board visibility flag, we may need to reset the auto-add board.
       */

      const state = getState();
      const queryArgs = selectListBoardsQueryArgs(state);
      const queryResult = boardsApi.endpoints.listAllBoards.select(queryArgs)(state);
      const autoAddBoardId = state.gallery.autoAddBoardId;

      if (!queryResult.data) {
        return;
      }

      if (!queryResult.data.find((board) => board.board_id === autoAddBoardId)) {
        dispatch(autoAddBoardIdChanged('none'));
        dispatch(boardIdSelected({ boardId: 'none' }));
        dispatch(galleryViewChanged('images'));
      }
    },
  });
};
