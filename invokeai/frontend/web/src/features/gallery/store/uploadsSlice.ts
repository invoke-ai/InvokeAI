import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';

import { RootState } from 'app/store/store';
import {
  receivedUploadImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { ImageDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';

export type UploadsImageDTO = Omit<ImageDTO, 'image_type'> & {
  image_type: 'uploads';
};

export const uploadsAdapter = createEntityAdapter<UploadsImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.created_at, a.created_at),
});

type AdditionalUploadsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
};

export const initialUploadsState =
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
    uploadUpserted: uploadsAdapter.upsertOne,
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
      const { page, pages } = action.payload;

      // We know these will all be of the uploads type, but it's not represented in the API types
      const items = action.payload.items as UploadsImageDTO[];

      uploadsAdapter.setMany(state, items);

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

export const { uploadUpserted } = uploadsSlice.actions;

export default uploadsSlice.reducer;
