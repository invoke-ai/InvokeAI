import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { ImagesService, PaginatedResults_ImageDTO_ } from 'services/api';

export const IMAGES_PER_PAGE = 20;

type ReceivedResultImagesPageThunkConfig = {
  rejectValue: {
    error: unknown;
  };
};

export const receivedResultImagesPage = createAppAsyncThunk<
  PaginatedResults_ImageDTO_,
  void,
  ReceivedResultImagesPageThunkConfig
>(
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

    return response;
  }
);

type ReceivedUploadImagesPageThunkConfig = {
  rejectValue: {
    error: unknown;
  };
};

export const receivedUploadImagesPage = createAppAsyncThunk<
  PaginatedResults_ImageDTO_,
  void,
  ReceivedUploadImagesPageThunkConfig
>(
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

    return response;
  }
);
