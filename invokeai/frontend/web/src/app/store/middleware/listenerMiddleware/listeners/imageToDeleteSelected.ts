import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import {
  imageDeletionConfirmed,
  imageToDeleteSelected,
  isModalOpenChanged,
  selectImageUsage,
} from 'features/imageDeletion/store/imageDeletionSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageToDeleteSelectedListener = () => {
  startAppListening({
    actionCreator: imageToDeleteSelected,
    effect: async (action, { dispatch, getState, condition }) => {
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
