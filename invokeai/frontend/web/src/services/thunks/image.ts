import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { ImagesService } from 'services/api';
import { getHeaders } from 'services/util/getHeaders';

const imagesLog = log.child({ namespace: 'image' });

type ImageReceivedArg = Parameters<(typeof ImagesService)['getImage']>[0];

/**
 * `ImagesService.getImage()` thunk
 */
export const imageReceived = createAppAsyncThunk(
  'api/imageReceived',
  async (arg: ImageReceivedArg) => {
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
  async (arg: ThumbnailReceivedArg) => {
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
  async (arg: ImageUploadedArg) => {
    const response = await ImagesService.uploadImage(arg);
    const { location } = getHeaders(response);

    imagesLog.info(
      { arg: '<Blob>', response, location },
      `Image uploaded (${response.image_name})`
    );

    return { response, location };
  }
);

type ImageDeletedArg = Parameters<(typeof ImagesService)['deleteImage']>[0];

/**
 * `ImagesService.deleteImage()` thunk
 */
export const imageDeleted = createAppAsyncThunk(
  'api/imageDeleted',
  async (arg: ImageDeletedArg) => {
    const response = await ImagesService.deleteImage(arg);

    imagesLog.info(
      { arg, response },
      `Image deleted (${arg.imageType} - ${arg.imageName})`
    );

    return response;
  }
);
