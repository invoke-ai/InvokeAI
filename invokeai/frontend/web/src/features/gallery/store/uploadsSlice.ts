import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/invokeai';

import { RootState } from 'app/store';
import {
  receivedUploadImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { deserializeImageField } from 'services/util/deserializeImageField';
import { deserializeImageResponse } from 'services/util/deserializeImageResponse';

export const uploadsAdapter = createEntityAdapter<Image>({
  selectId: (image) => image.name,
  sortComparer: (a, b) => b.metadata.timestamp - a.metadata.timestamp,
});

type AdditionalUploadsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
};

const uploadsSlice = createSlice({
  name: 'uploads',
  initialState: uploadsAdapter.getInitialState<AdditionalUploadsState>({
    page: 0,
    pages: 0,
    nextPage: 0,
    isLoading: false,
  }),
  reducers: {
    uploadAdded: uploadsAdapter.addOne,
  },
  extraReducers: (builder) => {
    builder.addCase(receivedUploadImagesPage.pending, (state) => {
      state.isLoading = true;
    });
    builder.addCase(receivedUploadImagesPage.fulfilled, (state, action) => {
      const { items, page, pages } = action.payload;

      const images = items.map((image) => deserializeImageResponse(image));

      uploadsAdapter.addMany(state, images);

      state.page = page;
      state.pages = pages;
      state.nextPage = items.length < IMAGES_PER_PAGE ? page : page + 1;
      state.isLoading = false;
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
