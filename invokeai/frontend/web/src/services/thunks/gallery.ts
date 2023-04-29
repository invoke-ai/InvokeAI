import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { ImagesService } from 'services/api';

export const IMAGES_PER_PAGE = 20;

const galleryLog = log.child({ namespace: 'gallery' });

export const receivedResultImagesPage = createAppAsyncThunk(
  'results/receivedResultImagesPage',
  async (_arg, { getState }) => {
    const response = await ImagesService.listImages({
      imageType: 'results',
      page: getState().results.nextPage,
      perPage: IMAGES_PER_PAGE,
    });

    galleryLog.info({ response }, 'Received page of results images');

    return response;
  }
);

export const receivedUploadImagesPage = createAppAsyncThunk(
  'uploads/receivedUploadImagesPage',
  async (_arg, { getState }) => {
    const response = await ImagesService.listImages({
      imageType: 'uploads',
      page: getState().uploads.nextPage,
      perPage: IMAGES_PER_PAGE,
    });

    galleryLog.info({ response }, 'Received page of uploads images');

    return response;
  }
);
