import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';
import { invocationComplete } from 'services/events/actions';

import { RootState } from 'app/store/store';
import {
  receivedResultImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { isImageOutput } from 'services/types/guards';
import {
  buildImageUrls,
  extractTimestampFromImageName,
} from 'services/util/deserializeImageField';
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
     * Invocation Complete
     */
    builder.addCase(invocationComplete, (state, action) => {
      const { data, shouldFetchImages } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      if (isImageOutput(result)) {
        const name = result.image.image_name;
        const type = result.image.image_type;

        // if we need to refetch, set URLs to placeholder for now
        const { url, thumbnail } = shouldFetchImages
          ? { url: '', thumbnail: '' }
          : buildImageUrls(type, name);

        const timestamp = extractTimestampFromImageName(name);

        const image: Image = {
          name,
          type,
          url,
          thumbnail,
          metadata: {
            created: timestamp,
            width: result.width,
            height: result.height,
            invokeai: {
              session_id: graph_execution_state_id,
              ...(node ? { node } : {}),
            },
          },
        };

        resultsAdapter.setOne(state, image);
      }
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
     * Delete Image - FULFILLED
     */
    builder.addCase(imageDeleted.fulfilled, (state, action) => {
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
