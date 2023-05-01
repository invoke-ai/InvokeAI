import { isFulfilled, isRejected } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { clamp, isString } from 'lodash-es';
import { ImagesService } from 'services/api';
import { getHeaders } from 'services/util/getHeaders';

const imagesLog = log.child({ namespace: 'image' });

type ImageReceivedArg = Parameters<(typeof ImagesService)['getImage']>[0];

/**
 * `ImagesService.getImage()` thunk
 */
export const imageReceived = createAppAsyncThunk(
  'api/imageReceived',
  async (arg: ImageReceivedArg, _thunkApi) => {
    const response = await ImagesService.getImage(arg);

    imagesLog.info({ arg, response }, 'Received image');

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

    imagesLog.info({ arg, response }, 'Received thumbnail');

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

    imagesLog.info(
      { arg: '<Blob>', response, location },
      `Image uploaded (${response.image_name})`
    );

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
    const selectedImageName = getState().gallery.selectedImage?.name;

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

      if (newSelectedImageId) {
        dispatch(
          imageSelected({ name: newSelectedImageId as string, type: imageType })
        );
      } else {
        dispatch(imageSelected());
      }
    }

    const response = await ImagesService.deleteImage(arg);

    imagesLog.info(
      { arg, response },
      `Image deleted (${arg.imageType} - ${arg.imageName})`
    );

    return response;
  }
);
