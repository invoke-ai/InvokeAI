import { AnyAction, ThunkAction } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import * as InvokeAI from 'app/invokeai';
import { v4 as uuidv4 } from 'uuid';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { setInitialImage } from 'features/options/store/optionsSlice';
import { addImage } from '../gallerySlice';

type UploadImageConfig = {
  imageFile: File;
};

export const uploadImage =
  (
    config: UploadImageConfig
  ): ThunkAction<void, RootState, unknown, AnyAction> =>
  async (dispatch, getState) => {
    const { imageFile } = config;

    const state = getState() as RootState;

    const activeTabName = activeTabNameSelector(state);

    const formData = new FormData();

    formData.append('file', imageFile, imageFile.name);
    formData.append(
      'data',
      JSON.stringify({
        kind: 'init',
      })
    );

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      body: formData,
    });

    const image = (await response.json()) as InvokeAI.ImageUploadResponse;
    console.log(image)
    const newImage: InvokeAI.Image = {
      uuid: uuidv4(),
      category: 'user',
      ...image,
    };

    dispatch(addImage({ image: newImage, category: 'user' }));

    if (activeTabName === 'unifiedCanvas') {
      dispatch(setInitialCanvasImage(newImage));
    } else if (activeTabName === 'img2img') {
      dispatch(setInitialImage(newImage));
    }
  };
