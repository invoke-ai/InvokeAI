import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { ImagesService } from 'services/api';
import { getHeaders } from 'services/util/getHeaders';

const imagesLog = log.child({ namespace: 'image' });

type imageUrlsReceivedArg = Parameters<
  (typeof ImagesService)['getImageUrls']
>[0];

/**
 * `ImagesService.getImageUrls()` thunk
 */
export const imageUrlsReceived = createAppAsyncThunk(
  'api/imageUrlsReceived',
  async (arg: imageUrlsReceivedArg) => {
    const response = await ImagesService.getImageUrls(arg);
    imagesLog.info({ arg, response }, 'Received image urls');
    return response;
  }
);

type imageMetadataReceivedArg = Parameters<
  (typeof ImagesService)['getImageMetadata']
>[0];

/**
 * `ImagesService.getImageUrls()` thunk
 */
export const imageMetadataReceived = createAppAsyncThunk(
  'api/imageMetadataReceived',
  async (arg: imageMetadataReceivedArg) => {
    const response = await ImagesService.getImageMetadata(arg);
    imagesLog.info({ arg, response }, 'Received image record');
    return response;
  }
);

type ImageUploadedArg = Parameters<(typeof ImagesService)['uploadImage']>[0] & {
  // extra arg to determine post-upload actions - we check for this when the image is uploaded
  // to determine if we should set the init image
  activeTabName?: InvokeTabName;
};

/**
 * `ImagesService.uploadImage()` thunk
 */
export const imageUploaded = createAppAsyncThunk(
  'api/imageUploaded',
  async (arg: ImageUploadedArg) => {
    // strip out `activeTabName` from arg - the route does not need it
    const { activeTabName, ...rest } = arg;
    const response = await ImagesService.uploadImage(rest);
    const { location } = getHeaders(response);

    imagesLog.debug({ arg: '<Blob>', response, location }, 'Image uploaded');

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

    imagesLog.debug({ arg, response }, 'Image deleted');

    return response;
  }
);
