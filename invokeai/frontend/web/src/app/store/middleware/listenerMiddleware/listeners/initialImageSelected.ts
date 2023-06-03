import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { t } from 'i18next';
import { addToast } from 'features/system/store/systemSlice';
import { startAppListening } from '..';
import {
  initialImageSelected,
  isImageDTO,
} from 'features/parameters/store/actions';
import { makeToast } from 'app/components/Toaster';
import { selectImagesById } from 'features/gallery/store/imagesSlice';

export const addInitialImageSelectedListener = () => {
  startAppListening({
    actionCreator: initialImageSelected,
    effect: (action, { getState, dispatch }) => {
      if (!action.payload) {
        dispatch(
          addToast(
            makeToast({ title: t('toast.imageNotLoadedDesc'), status: 'error' })
          )
        );
        return;
      }

      if (isImageDTO(action.payload)) {
        dispatch(initialImageChanged(action.payload));
        dispatch(addToast(makeToast(t('toast.sentToImageToImage'))));
        return;
      }

      const imageName = action.payload;
      const image = selectImagesById(getState(), imageName);

      if (!image) {
        dispatch(
          addToast(
            makeToast({ title: t('toast.imageNotLoadedDesc'), status: 'error' })
          )
        );
        return;
      }

      dispatch(initialImageChanged(image));
      dispatch(addToast(makeToast(t('toast.sentToImageToImage'))));
    },
  });
};
