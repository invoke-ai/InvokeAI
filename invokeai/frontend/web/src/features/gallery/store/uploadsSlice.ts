import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/invokeai';

import { RootState } from 'app/store';
import {
  receivedUploadImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { imageUploaded } from 'services/thunks/image';
import { deserializeImageField } from 'services/util/deserializeImageField';
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
      const { image_name, image_url, image_type, metadata, thumbnail_url } =
        response;

      const uploadedImage: Image = {
        name: image_name,
        url: image_url,
        thumbnail: thumbnail_url,
        type: 'uploads',
        metadata,
      };

      uploadsAdapter.addOne(state, uploadedImage);
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
