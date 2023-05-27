import { createAppAsyncThunk } from 'app/store/storeUtils';
import { ImagesService, PaginatedResults_ImageDTO_ } from 'services/api';

export const IMAGES_PER_PAGE = 20;

type ReceivedResultImagesPageThunkConfig = {
  rejectValue: {
    error: unknown;
  };
};

export const receivedGalleryImages = createAppAsyncThunk<
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
      excludeCategories: ['user'],
      isIntermediate: false,
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

export const receivedUploadImages = createAppAsyncThunk<
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
      includeCategories: ['user'],
      isIntermediate: false,
      page: nextPage + pageOffset,
      perPage: IMAGES_PER_PAGE,
    });

    return response;
  }
);
