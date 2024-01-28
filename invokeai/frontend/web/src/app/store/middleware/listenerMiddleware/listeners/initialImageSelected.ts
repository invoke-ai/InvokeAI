import { initialImageSelected } from 'features/parameters/store/actions';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';

import { startAppListening } from '..';

export const addInitialImageSelectedListener = () => {
  startAppListening({
    actionCreator: initialImageSelected,
    effect: (action, { dispatch }) => {
      if (!action.payload) {
        dispatch(addToast(makeToast({ title: t('toast.imageNotLoadedDesc'), status: 'error' })));
        return;
      }

      dispatch(initialImageChanged(action.payload));
      dispatch(addToast(makeToast(t('toast.sentToImageToImage'))));
    },
  });
};
