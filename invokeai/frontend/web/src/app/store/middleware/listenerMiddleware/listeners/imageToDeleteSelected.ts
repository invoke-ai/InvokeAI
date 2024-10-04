import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { selectImageUsage } from 'features/deleteImageModal/store/selectors';
import { imagesToDeleteSelected, isModalOpenChanged } from 'features/deleteImageModal/store/slice';

export const addImageToDeleteSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: imagesToDeleteSelected,
    effect: (action, { dispatch, getState }) => {
      const imageDTOs = action.payload;
      const state = getState();
      const { shouldConfirmOnDelete } = state.system;
      const imagesUsage = selectImageUsage(getState());

      const isImageInUse =
        imagesUsage.some((i) => i.isRasterLayerImage) ||
        imagesUsage.some((i) => i.isControlLayerImage) ||
        imagesUsage.some((i) => i.isReferenceImage) ||
        imagesUsage.some((i) => i.isInpaintMaskImage) ||
        imagesUsage.some((i) => i.isUpscaleImage) ||
        imagesUsage.some((i) => i.isNodesImage) ||
        imagesUsage.some((i) => i.isRegionalGuidanceImage);

      if (shouldConfirmOnDelete || isImageInUse) {
        dispatch(isModalOpenChanged(true));
        return;
      }

      dispatch(imageDeletionConfirmed({ imageDTOs, imagesUsage }));
    },
  });
};
