import { isFulfilled } from '@reduxjs/toolkit';
import { createAppAsyncThunk } from 'app/storeUtils';
import { ImagesService } from 'services/api';
import { getHeaders } from 'services/util/getHeaders';

type GetImageArg = Parameters<(typeof ImagesService)['getImage']>[0];

/**
 * `ImagesService.getImage()` thunk
 */
export const getImage = createAppAsyncThunk(
  'api/getImage',
  async (arg: GetImageArg, _thunkApi) => {
    const response = await ImagesService.getImage(arg);
    return response;
  }
);

type UploadImageArg = Parameters<(typeof ImagesService)['uploadImage']>[0];

/**
 * `ImagesService.uploadImage()` thunk
 */
export const uploadImage = createAppAsyncThunk(
  'api/uploadImage',
  async (arg: UploadImageArg, _thunkApi) => {
    const response = await ImagesService.uploadImage(arg);
    const { location } = getHeaders(response);
    return location;
  }
);

/**
 * Function to check if an action is a fulfilled `ImagesService.uploadImage()` thunk
 */
export const isFulfilledUploadImage = isFulfilled(uploadImage);
