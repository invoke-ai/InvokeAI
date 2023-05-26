import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { ImagesService } from 'services/api';

export const IMAGES_PER_PAGE = 20;

const galleryLog = log.child({ namespace: 'gallery' });

export const receivedResultImagesPage = createAppAsyncThunk(
  'results/receivedResultImagesPage',
  async (_arg, { getState, rejectWithValue }) => {
    const { page, pages, nextPage, upsertedImageCount } = getState().results;

    // If many images have been upserted, we need to offset the page number
    // TODO: add an offset param to the list images endpoint
    const pageOffset = Math.floor(upsertedImageCount / IMAGES_PER_PAGE);

    const response = await ImagesService.listImagesWithMetadata({
      imageType: 'results',
      imageCategory: 'general',
      page: nextPage + pageOffset,
      perPage: IMAGES_PER_PAGE,
    });

    galleryLog.info({ response }, `Received ${response.items.length} results`);

    return response;
  }
);

export const receivedUploadImagesPage = createAppAsyncThunk(
  'uploads/receivedUploadImagesPage',
  async (_arg, { getState, rejectWithValue }) => {
    const { page, pages, nextPage, upsertedImageCount } = getState().uploads;

    // If many images have been upserted, we need to offset the page number
    // TODO: add an offset param to the list images endpoint
    const pageOffset = Math.floor(upsertedImageCount / IMAGES_PER_PAGE);

    const response = await ImagesService.listImagesWithMetadata({
      imageType: 'uploads',
      imageCategory: 'general',
      page: nextPage + pageOffset,
      perPage: IMAGES_PER_PAGE,
    });

    galleryLog.info({ response }, `Received ${response.items.length} uploads`);

    return response;
  }
);
