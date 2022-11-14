import { createAsyncThunk } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import * as InvokeAI from 'app/invokeai';
import { v4 as uuidv4 } from 'uuid';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

export const uploadImage = createAsyncThunk(
  'gallery/uploadImage',
  async (
    args: {
      imageFile: File;
    },
    thunkAPI
  ) => {
    const { imageFile } = args;

    const { getState } = thunkAPI;

    const state = getState() as RootState;

    const activeTabName = activeTabNameSelector(state);

    const formData = new FormData();

    formData.append('file', imageFile, imageFile.name);
    formData.append('kind', 'init');

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      body: formData,
    });

    const { image } = (await response.json()) as InvokeAI.ImageUploadResponse;

    const newImage: InvokeAI.Image = {
      uuid: uuidv4(),
      category: 'user',
      ...image,
    };

    return {
      image: newImage,
      kind: 'init',
      activeTabName,
    };
  }
);
