import {
  PayloadAction,
  createEntityAdapter,
  createSlice,
} from '@reduxjs/toolkit';

import { RootState } from 'app/store/store';
import { receivedUploadImages, IMAGES_PER_PAGE } from 'services/thunks/gallery';
import { ImageDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';

export type UploadsImageDTO = Omit<
  ImageDTO,
  'image_origin' | 'image_category'
> & {
  image_origin: 'external';
  image_category: 'user';
};

export const uploadsAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.created_at, a.created_at),
});

type AdditionalUploadsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
  upsertedImageCount: number;
};

export const initialUploadsState =
  uploadsAdapter.getInitialState<AdditionalUploadsState>({
    page: 0,
    pages: 0,
    nextPage: 0,
    isLoading: false,
    upsertedImageCount: 0,
  });

export type UploadsState = typeof initialUploadsState;

const uploadsSlice = createSlice({
  name: 'uploads',
  initialState: initialUploadsState,
  reducers: {
    uploadUpserted: (state, action: PayloadAction<ImageDTO>) => {
      uploadsAdapter.upsertOne(state, action.payload);
      state.upsertedImageCount += 1;
    },
    uploadRemoved: (state, action: PayloadAction<string>) => {
      uploadsAdapter.removeOne(state, action.payload);
    },
  },
  extraReducers: (builder) => {
    /**
     * Received Upload Images Page - PENDING
     */
    builder.addCase(receivedUploadImages.pending, (state) => {
      state.isLoading = true;
    });

    /**
     * Received Upload Images Page - FULFILLED
     */
    builder.addCase(receivedUploadImages.fulfilled, (state, action) => {
      const { page, pages } = action.payload;

      // We know these will all be of the uploads type, but it's not represented in the API types
      const items = action.payload.items;

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

export const { uploadUpserted, uploadRemoved } = uploadsSlice.actions;

export default uploadsSlice.reducer;
