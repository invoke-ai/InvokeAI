import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/invokeai';

import { RootState } from 'app/store';
import { getNextUploadsPage, IMAGES_PER_PAGE } from 'services/thunks/gallery';
import { processImageField } from 'services/util/processImageField';

export const uploadsAdapter = createEntityAdapter<Image>({
  selectId: (image) => image.name,
  sortComparer: (a, b) => b.timestamp - a.timestamp,
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
    builder.addCase(getNextUploadsPage.pending, (state) => {
      state.isLoading = true;
    });
    builder.addCase(getNextUploadsPage.fulfilled, (state, action) => {
      const { items, page, pages } = action.payload;

      const images = items.map((image) => processImageField(image));

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
