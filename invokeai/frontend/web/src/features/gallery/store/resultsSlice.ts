import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';

import { RootState } from 'app/store/store';
import {
  receivedResultImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import {
  imageDeleted,
  imageReceived,
  thumbnailReceived,
} from 'services/thunks/image';

export const resultsAdapter = createEntityAdapter<Image>({
  selectId: (image) => image.name,
  sortComparer: (a, b) => b.metadata.created - a.metadata.created,
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
      const { items, page, pages } = action.payload;

      const resultImages = items.map((image) =>
        deserializeImageResponse(image)
      );

      resultsAdapter.setMany(state, resultImages);

      state.page = page;
      state.pages = pages;
      state.nextPage = items.length < IMAGES_PER_PAGE ? page : page + 1;
      state.isLoading = false;
    });

    /**
     * Image Received - FULFILLED
     */
    builder.addCase(imageReceived.fulfilled, (state, action) => {
      const { imagePath } = action.payload;
      const { imageName } = action.meta.arg;

      resultsAdapter.updateOne(state, {
        id: imageName,
        changes: {
          url: imagePath,
        },
      });
    });

    /**
     * Thumbnail Received - FULFILLED
     */
    builder.addCase(thumbnailReceived.fulfilled, (state, action) => {
      const { thumbnailPath } = action.payload;
      const { thumbnailName } = action.meta.arg;

      resultsAdapter.updateOne(state, {
        id: thumbnailName,
        changes: {
          thumbnail: thumbnailPath,
        },
      });
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
