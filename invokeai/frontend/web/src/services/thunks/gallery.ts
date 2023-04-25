import { createAppAsyncThunk } from 'app/storeUtils';
import { ImagesService } from 'services/api';

export const IMAGES_PER_PAGE = 20;

export const receivedResultImagesPage = createAppAsyncThunk(
  'results/receivedResultImagesPage',
  async (_arg, { getState }) => {
    const response = await ImagesService.listImages({
      imageType: 'results',
      page: getState().results.nextPage,
      perPage: IMAGES_PER_PAGE,
    });

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

    return response;
  }
);
