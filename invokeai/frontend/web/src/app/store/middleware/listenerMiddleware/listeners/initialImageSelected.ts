import { makeToast } from 'app/components/Toaster';
import { initialImageSelected } from 'features/parameters/store/actions';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { startAppListening } from '..';

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

      dispatch(initialImageChanged(action.payload));
      dispatch(addToast(makeToast(t('toast.sentToImageToImage'))));
    },
  });
};
