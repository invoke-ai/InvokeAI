import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  receivedResultImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import {
  imageDeleted,
  imageMetadataReceived,
  imageUrlsReceived,
} from 'services/thunks/image';
import { ImageDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';

export type ResultsImageDTO = Omit<ImageDTO, 'image_type'> & {
  image_type: 'results';
};

export const resultsAdapter = createEntityAdapter<ResultsImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.created_at, a.created_at),
});

type AdditionalResultsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
};

export const initialResultsState =
  resultsAdapter.getInitialState<AdditionalResultsState>({
    page: 0,
    pages: 0,
    isLoading: false,
    nextPage: 0,
  });

export type ResultsState = typeof initialResultsState;

const resultsSlice = createSlice({
  name: 'results',
  initialState: initialResultsState,
  reducers: {
    resultAdded: resultsAdapter.upsertOne,
  },
  extraReducers: (builder) => {
    /**
     * Received Result Images Page - PENDING
     */
    builder.addCase(receivedResultImagesPage.pending, (state) => {
      state.isLoading = true;
    });

    /**
     * Received Result Images Page - FULFILLED
     */
    builder.addCase(receivedResultImagesPage.fulfilled, (state, action) => {
      const { page, pages } = action.payload;

      // We know these will all be of the results type, but it's not represented in the API types
      const items = action.payload.items as ResultsImageDTO[];

      resultsAdapter.setMany(state, items);

      state.page = page;
      state.pages = pages;
      state.nextPage = items.length < IMAGES_PER_PAGE ? page : page + 1;
      state.isLoading = false;
    });

    /**
     * Image Metadata Received - FULFILLED
     */
    builder.addCase(imageMetadataReceived.fulfilled, (state, action) => {
      const { image_type } = action.payload;

      if (image_type === 'results') {
        resultsAdapter.upsertOne(state, action.payload as ResultsImageDTO);
      }
    });

    /**
     * Image URLs Received - FULFILLED
     */
    builder.addCase(imageUrlsReceived.fulfilled, (state, action) => {
      const { image_name, image_type, image_url, thumbnail_url } =
        action.payload;

      if (image_type === 'results') {
        resultsAdapter.updateOne(state, {
          id: image_name,
          changes: {
            image_url: image_url,
            thumbnail_url: thumbnail_url,
          },
        });
      }
    });

    /**
     * Delete Image - PENDING
     * Pre-emptively remove the image from the gallery
     */
    builder.addCase(imageDeleted.pending, (state, action) => {
      const { imageType, imageName } = action.meta.arg;

      if (imageType === 'results') {
        resultsAdapter.removeOne(state, imageName);
      }
    });
  },
});

export const {
  selectAll: selectResultsAll,
  selectById: selectResultsById,
  selectEntities: selectResultsEntities,
  selectIds: selectResultsIds,
  selectTotal: selectResultsTotal,
} = resultsAdapter.getSelectors<RootState>((state) => state.results);

export const { resultAdded } = resultsSlice.actions;

export default resultsSlice.reducer;
