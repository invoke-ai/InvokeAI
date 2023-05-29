import {
  PayloadAction,
  createEntityAdapter,
  createSelector,
  createSlice,
} from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { ImageCategory, ImageDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';
import { isString, keyBy } from 'lodash-es';
import { receivedPageOfImages } from 'services/thunks/image';

export const imagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.created_at, a.created_at),
});

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = [
  'control',
  'mask',
  'user',
  'other',
];

type AdditionaImagesState = {
  offset: number;
  limit: number;
  total: number;
  isLoading: boolean;
  categories: ImageCategory[];
};

export const initialImagesState =
  imagesAdapter.getInitialState<AdditionaImagesState>({
    offset: 0,
    limit: 0,
    total: 0,
    isLoading: false,
    categories: IMAGE_CATEGORIES,
  });

export type ImagesState = typeof initialImagesState;

const imagesSlice = createSlice({
  name: 'images',
  initialState: initialImagesState,
  reducers: {
    imageUpserted: (state, action: PayloadAction<ImageDTO>) => {
      imagesAdapter.upsertOne(state, action.payload);
    },
    imageRemoved: (state, action: PayloadAction<string | ImageDTO>) => {
      if (isString(action.payload)) {
        imagesAdapter.removeOne(state, action.payload);
        return;
      }

      imagesAdapter.removeOne(state, action.payload.image_name);
    },
    imageCategoriesChanged: (state, action: PayloadAction<ImageCategory[]>) => {
      state.categories = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedPageOfImages.pending, (state) => {
      state.isLoading = true;
    });
    builder.addCase(receivedPageOfImages.rejected, (state) => {
      state.isLoading = false;
    });
    builder.addCase(receivedPageOfImages.fulfilled, (state, action) => {
      state.isLoading = false;
      const { items, offset, limit, total } = action.payload;
      state.offset = offset;
      state.limit = limit;
      state.total = total;
      imagesAdapter.upsertMany(state, items);
    });
  },
});

export const {
  selectAll: selectImagesAll,
  selectById: selectImagesById,
  selectEntities: selectImagesEntities,
  selectIds: selectImagesIds,
  selectTotal: selectImagesTotal,
} = imagesAdapter.getSelectors<RootState>((state) => state.images);

export const { imageUpserted, imageRemoved, imageCategoriesChanged } =
  imagesSlice.actions;

export default imagesSlice.reducer;

export const selectFilteredImagesAsArray = createSelector(
  (state: RootState) => state,
  (state) => {
    const {
      images: { categories },
    } = state;

    return selectImagesAll(state).filter((i) =>
      categories.includes(i.image_category)
    );
  }
);

export const selectFilteredImagesAsObject = createSelector(
  (state: RootState) => state,
  (state) => {
    const {
      images: { categories },
    } = state;

    return keyBy(
      selectImagesAll(state).filter((i) =>
        categories.includes(i.image_category)
      ),
      'image_name'
    );
  }
);

export const selectFilteredImagesIds = createSelector(
  (state: RootState) => state,
  (state) => {
    const {
      images: { categories },
    } = state;

    return selectImagesAll(state)
      .filter((i) => categories.includes(i.image_category))
      .map((i) => i.image_name);
  }
);
