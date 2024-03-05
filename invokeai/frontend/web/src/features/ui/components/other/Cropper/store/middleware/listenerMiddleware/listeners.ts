import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { cropperImageToGallery } from 'features/ui/components/other/Cropper/store/actions';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

export const addCropperImageToGalleryListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: cropperImageToGallery,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();

      const imageBlob = await fetch(action.payload.id).then((res) => res.blob());

      const { autoAddBoardId } = state.gallery;

      await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([imageBlob], 'croppedImage.png', {
            type: 'image/png',
          }),
          image_category: 'general',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: false,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: t('cropper.croppedImageSaved') },
          },
        })
      ).unwrap();
    },
  });
};
