import { isFulfilled, isRejected } from '@reduxjs/toolkit';
import { createAppAsyncThunk } from 'app/storeUtils';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { clamp } from 'lodash';
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

type ThumbnailReceivedArg = Parameters<
  (typeof ImagesService)['getThumbnail']
>[0];

/**
 * `ImagesService.getThumbnail()` thunk
 */
export const thumbnailReceived = createAppAsyncThunk(
  'api/thumbnailReceived',
  async (arg: ThumbnailReceivedArg, _thunkApi) => {
    const response = await ImagesService.getThumbnail(arg);
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
    return { response, location };
  }
);

/**
 * Function to check if an action is a fulfilled `ImagesService.uploadImage()` thunk
 */
export const isFulfilledImageUploadedAction = isFulfilled(imageUploaded);

type ImageDeletedArg = Parameters<(typeof ImagesService)['deleteImage']>[0];

/**
 * `ImagesService.deleteImage()` thunk
 */
export const imageDeleted = createAppAsyncThunk(
  'api/imageDeleted',
  async (arg: ImageDeletedArg, { getState, dispatch }) => {
    const { imageType, imageName } = arg;

    if (imageType !== 'uploads' && imageType !== 'results') {
      return;
    }

    // TODO: move this logic to another thunk?
    // Determine which image should replace the deleted image, if the deleted image is the selected image.
    // Unfortunately, we have to do this here, because the resultsSlice and uploadsSlice cannot change
    // the selected image.
    const selectedImageName = getState().gallery.selectedImageName;

    if (selectedImageName === imageName) {
      const allIds = getState()[imageType].ids;

      const deletedImageIndex = allIds.findIndex(
        (result) => result.toString() === imageName
      );

      const filteredIds = allIds.filter((id) => id.toString() !== imageName);

      const newSelectedImageIndex = clamp(
        deletedImageIndex,
        0,
        filteredIds.length - 1
      );

      const newSelectedImageId = filteredIds[newSelectedImageIndex];

      dispatch(
        imageSelected(newSelectedImageId ? newSelectedImageId.toString() : '')
      );
    }

    const response = await ImagesService.deleteImage(arg);

    return response;
  }
);
