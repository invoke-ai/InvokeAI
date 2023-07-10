import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { requestedBoardImagesDeletion } from 'features/gallery/store/actions';
import {
  imageSelected,
  imagesRemoved,
  selectImagesAll,
  selectImagesById,
} from 'features/gallery/store/gallerySlice';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { LIST_TAG, api } from 'services/api';
import { startAppListening } from '..';
import { boardsApi } from '../../../../../services/api/endpoints/boards';

export const addRequestedBoardImageDeletionListener = () => {
  startAppListening({
    actionCreator: requestedBoardImagesDeletion,
    effect: async (action, { dispatch, getState, condition }) => {
      const { board, imagesUsage } = action.payload;

      const { board_id } = board;

      const state = getState();
      const selectedImageName =
        state.gallery.selection[state.gallery.selection.length - 1];

      const selectedImage = selectedImageName
        ? selectImagesById(state, selectedImageName)
        : undefined;

      if (selectedImage && selectedImage.board_id === board_id) {
        dispatch(imageSelected(null));
      }

      // We need to reset the features where the board images are in use - none of these work if their image(s) don't exist

      if (imagesUsage.isCanvasImage) {
        dispatch(resetCanvas());
      }

      if (imagesUsage.isControlNetImage) {
        dispatch(controlNetReset());
      }

      if (imagesUsage.isInitialImage) {
        dispatch(clearInitialImage());
      }

      if (imagesUsage.isNodesImage) {
        dispatch(nodeEditorReset());
      }

      // Preemptively remove from gallery
      const images = selectImagesAll(state).reduce((acc: string[], img) => {
        if (img.board_id === board_id) {
          acc.push(img.image_name);
        }
        return acc;
      }, []);
      dispatch(imagesRemoved(images));

      // Delete from server
      dispatch(boardsApi.endpoints.deleteBoardAndImages.initiate(board_id));
      const result =
        boardsApi.endpoints.deleteBoardAndImages.select(board_id)(state);
      const { isSuccess } = result;

      // Wait for successful deletion, then trigger boards to re-fetch
      const wasBoardDeleted = await condition(() => !!isSuccess, 30000);

      if (wasBoardDeleted) {
        dispatch(
          api.util.invalidateTags([
            { type: 'Board', id: board_id },
            { type: 'Image', id: LIST_TAG },
          ])
        );
      }
    },
  });
};
