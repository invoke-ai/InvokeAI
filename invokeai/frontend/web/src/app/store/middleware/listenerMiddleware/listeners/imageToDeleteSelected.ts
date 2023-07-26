import { imageDeletionConfirmed } from 'features/imageDeletion/store/actions';
import { selectImageUsage } from 'features/imageDeletion/store/imageDeletionSelectors';
import {
  imageToDeleteSelected,
  isModalOpenChanged,
} from 'features/imageDeletion/store/imageDeletionSlice';
import { startAppListening } from '..';

export const addImageToDeleteSelectedListener = () => {
  startAppListening({
    actionCreator: imageToDeleteSelected,
    effect: async (action, { dispatch, getState }) => {
      const imageDTO = action.payload;
      const state = getState();
      const { shouldConfirmOnDelete } = state.system;
      const imageUsage = selectImageUsage(getState());

      if (!imageUsage) {
        // should never happen
        return;
      }

      const isImageInUse =
        imageUsage.isCanvasImage ||
        imageUsage.isInitialImage ||
        imageUsage.isControlNetImage ||
        imageUsage.isNodesImage;

      if (shouldConfirmOnDelete || isImageInUse) {
        dispatch(isModalOpenChanged(true));
        return;
      }

      dispatch(imageDeletionConfirmed({ imageDTO, imageUsage }));
    },
  });
};
