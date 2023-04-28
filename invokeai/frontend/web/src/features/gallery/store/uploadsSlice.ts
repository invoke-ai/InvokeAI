import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/invokeai';

import { RootState } from 'app/store';
import {
  receivedUploadImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { imageDeleted, imageUploaded } from 'services/thunks/image';
import { deserializeImageResponse } from 'services/util/deserializeImageResponse';

export const uploadsAdapter = createEntityAdapter<Image>({
  selectId: (image) => image.name,
  sortComparer: (a, b) => b.metadata.created - a.metadata.created,
});

type AdditionalUploadsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
};

const initialUploadsState =
  uploadsAdapter.getInitialState<AdditionalUploadsState>({
    page: 0,
    pages: 0,
    nextPage: 0,
    isLoading: false,
  });

export type UploadsState = typeof initialUploadsState;

const uploadsSlice = createSlice({
  name: 'uploads',
  initialState: initialUploadsState,
  reducers: {
    uploadAdded: uploadsAdapter.addOne,
  },
  extraReducers: (builder) => {
    /**
     * Received Upload Images Page - PENDING
     */
    builder.addCase(receivedUploadImagesPage.pending, (state) => {
      state.isLoading = true;
    });

    /**
     * Received Upload Images Page - FULFILLED
     */
    builder.addCase(receivedUploadImagesPage.fulfilled, (state, action) => {
      const { items, page, pages } = action.payload;

      const images = items.map((image) => deserializeImageResponse(image));

      uploadsAdapter.addMany(state, images);

      state.page = page;
      state.pages = pages;
      state.nextPage = items.length < IMAGES_PER_PAGE ? page : page + 1;
      state.isLoading = false;
    });

    /**
     * Upload Image - FULFILLED
     */
    builder.addCase(imageUploaded.fulfilled, (state, action) => {
      const { location, response } = action.payload;

      const uploadedImage = deserializeImageResponse(response);

      uploadsAdapter.addOne(state, uploadedImage);
    });

    /**
     * Delete Image - FULFILLED
     */
    builder.addCase(imageDeleted.fulfilled, (state, action) => {
      const { imageType, imageName } = action.meta.arg;

      if (imageType === 'uploads') {
        uploadsAdapter.removeOne(state, imageName);
      }
    });
  },
});

export const {
  selectAll: selectUploadsAll,
  selectById: selectUploadsById,
  selectEntities: selectUploadsEntities,
  selectIds: selectUploadsIds,
  selectTotal: selectUploadsTotal,
} = uploadsAdapter.getSelectors<RootState>((state) => state.uploads);

export const { uploadAdded } = uploadsSlice.actions;

export default uploadsSlice.reducer;
