import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { selectResultsById } from 'features/gallery/store/resultsSlice';
import { selectUploadsById } from 'features/gallery/store/uploadsSlice';
import { t } from 'i18next';
import { addToast } from 'features/system/store/systemSlice';
import { startAppListening } from '..';
import {
  initialImageSelected,
  isImageDTO,
} from 'features/parameters/store/actions';
import { makeToast } from 'app/components/Toaster';
import { ImageDTO } from 'services/api';

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

      const { image_name, image_type } = action.payload;

      let image: ImageDTO | undefined;
      const state = getState();

      if (image_type === 'results') {
        image = selectResultsById(state, image_name);
      } else if (image_type === 'uploads') {
        image = selectUploadsById(state, image_name);
      }

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
