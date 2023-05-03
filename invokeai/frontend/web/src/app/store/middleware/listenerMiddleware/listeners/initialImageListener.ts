import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { Image, isInvokeAIImage } from 'app/types/invokeai';
import { selectResultsById } from 'features/gallery/store/resultsSlice';
import { selectUploadsById } from 'features/gallery/store/uploadsSlice';
import { makeToast } from 'features/system/hooks/useToastWatcher';
import { t } from 'i18next';
import { addToast } from 'features/system/store/systemSlice';
import { AnyAction, ListenerEffect } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store/store';

export const initialImageListener: ListenerEffect<
  AnyAction,
  RootState,
  AppDispatch
> = (action, { getState, dispatch }) => {
  if (!action.payload) {
    dispatch(
      addToast(
        makeToast({ title: t('toast.imageNotLoadedDesc'), status: 'error' })
      )
    );
    return;
  }

  if (isInvokeAIImage(action.payload)) {
    dispatch(initialImageChanged(action.payload));
    dispatch(addToast(makeToast(t('toast.sentToImageToImage'))));
    return;
  }

  const { name, type } = action.payload;

  let image: Image | undefined;
  const state = getState();

  if (type === 'results') {
    image = selectResultsById(state, name);
  } else if (type === 'uploads') {
    image = selectUploadsById(state, name);
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
};
