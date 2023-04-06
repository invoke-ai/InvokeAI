import { isFulfilled } from '@reduxjs/toolkit';
import { createAppAsyncThunk } from 'app/storeUtils';
import { ImagesService } from 'services/api';
import { getHeaders } from 'services/util/getHeaders';

type ImageReceivedArg = Parameters<(typeof ImagesService)['getImage']>[0];

/**
 * `ImagesService.getImage()` thunk
 */
export const imageReceived = createAppAsyncThunk(
  'api/imageReceived',
  async (arg: ImageReceivedArg, _thunkApi) => {
    const response = await ImagesService.getImage(arg);
    return response;
  }
);

type ImageUploadedArg = Parameters<(typeof ImagesService)['uploadImage']>[0];

/**
 * `ImagesService.uploadImage()` thunk
 */
export const imageUploaded = createAppAsyncThunk(
  'api/imageUploaded',
  async (arg: ImageUploadedArg, _thunkApi) => {
    const response = await ImagesService.uploadImage(arg);
    const { location } = getHeaders(response);
    return location;
  }
);

/**
 * Function to check if an action is a fulfilled `ImagesService.uploadImage()` thunk
 */
export const isFulfilledImageUploadedAction = isFulfilled(imageUploaded);
