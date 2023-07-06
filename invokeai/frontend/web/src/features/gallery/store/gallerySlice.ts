import type { PayloadAction, Update } from '@reduxjs/toolkit';
import {
  createEntityAdapter,
  createSelector,
  createSlice,
} from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { dateComparator } from 'common/util/dateComparator';
import { keyBy, uniq } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import {
  imageUrlsReceived,
  receivedPageOfImages,
} from 'services/api/thunks/image';
import { ImageCategory, ImageDTO } from 'services/api/types';

export const imagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.updated_at, a.updated_at),
});

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = [
  'control',
  'mask',
  'user',
  'other',
];

export const INITIAL_IMAGE_LIMIT = 100;
export const IMAGE_LIMIT = 20;

type AdditionaGalleryState = {
  offset: number;
  limit: number;
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  categories: ImageCategory[];
  selectedBoardId?: string;
  selection: string[];
  shouldAutoSwitch: boolean;
  galleryImageMinimumWidth: number;
  galleryView: 'images' | 'assets';
  isInitialized: boolean;
};

export const initialGalleryState =
  imagesAdapter.getInitialState<AdditionaGalleryState>({
    offset: 0,
    limit: 0,
    total: 0,
    isLoading: true,
    isFetching: true,
    categories: IMAGE_CATEGORIES,
    selection: [],
    shouldAutoSwitch: true,
    galleryImageMinimumWidth: 96,
    galleryView: 'images',
    isInitialized: false,
  });

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageUpserted: (state, action: PayloadAction<ImageDTO>) => {
      imagesAdapter.upsertOne(state, action.payload);
      if (
        state.shouldAutoSwitch &&
        action.payload.image_category === 'general'
      ) {
        state.selection = [action.payload.image_name];
        state.galleryView = 'images';
        state.categories = IMAGE_CATEGORIES;
      }
    },
    imageUpdatedOne: (state, action: PayloadAction<Update<ImageDTO>>) => {
      imagesAdapter.updateOne(state, action.payload);
    },
    imageUpdatedMany: (state, action: PayloadAction<Update<ImageDTO>[]>) => {
      imagesAdapter.updateMany(state, action.payload);
    },
    imageRemoved: (state, action: PayloadAction<string>) => {
      imagesAdapter.removeOne(state, action.payload);
    },
    imagesRemoved: (state, action: PayloadAction<string[]>) => {
      imagesAdapter.removeMany(state, action.payload);
    },
    imageCategoriesChanged: (state, action: PayloadAction<ImageCategory[]>) => {
      state.categories = action.payload;
    },
    imageRangeEndSelected: (state, action: PayloadAction<string>) => {
      const rangeEndImageName = action.payload;
      const lastSelectedImage = state.selection[state.selection.length - 1];

      const filteredImages = selectFilteredImagesLocal(state);

      const lastClickedIndex = filteredImages.findIndex(
        (n) => n.image_name === lastSelectedImage
      );

      const currentClickedIndex = filteredImages.findIndex(
        (n) => n.image_name === rangeEndImageName
      );

      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);

        const imagesToSelect = filteredImages
          .slice(start, end + 1)
          .map((i) => i.image_name);

        state.selection = uniq(state.selection.concat(imagesToSelect));
      }
    },
    imageSelectionToggled: (state, action: PayloadAction<string>) => {
      if (
        state.selection.includes(action.payload) &&
        state.selection.length > 1
      ) {
        state.selection = state.selection.filter(
          (imageName) => imageName !== action.payload
        );
      } else {
        state.selection = uniq(state.selection.concat(action.payload));
      }
    },
    imageSelected: (state, action: PayloadAction<string | null>) => {
      state.selection = action.payload
        ? [action.payload]
        : [String(state.ids[0])];
    },
    shouldAutoSwitchChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoSwitch = action.payload;
    },
    setGalleryImageMinimumWidth: (state, action: PayloadAction<number>) => {
      state.galleryImageMinimumWidth = action.payload;
    },
    setGalleryView: (state, action: PayloadAction<'images' | 'assets'>) => {
      state.galleryView = action.payload;
    },
    boardIdSelected: (state, action: PayloadAction<string | undefined>) => {
      state.selectedBoardId = action.payload;
    },
    isLoadingChanged: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedPageOfImages.pending, (state) => {
      state.isFetching = true;
    });
    builder.addCase(receivedPageOfImages.rejected, (state) => {
      state.isFetching = false;
    });
    builder.addCase(receivedPageOfImages.fulfilled, (state, action) => {
      state.isFetching = false;
      const { board_id, categories, image_origin, is_intermediate } =
        action.meta.arg;

      const { items, offset, limit, total } = action.payload;

      imagesAdapter.upsertMany(state, items);

      if (state.selection.length === 0 && items.length) {
        state.selection = [items[0].image_name];
      }

      if (!categories?.includes('general') || board_id) {
        // need to skip updating the total images count if the images recieved were for a specific board
        // TODO: this doesn't work when on the Asset tab/category...
        return;
      }

      state.offset = offset;
      state.total = total;
    });
    builder.addCase(imageUrlsReceived.fulfilled, (state, action) => {
      const { image_name, image_url, thumbnail_url } = action.payload;

      imagesAdapter.updateOne(state, {
        id: image_name,
        changes: { image_url, thumbnail_url },
      });
    });
    builder.addMatcher(
      boardsApi.endpoints.deleteBoard.matchFulfilled,
      (state, action) => {
        if (action.meta.arg.originalArgs === state.selectedBoardId) {
          state.selectedBoardId = undefined;
        }
      }
    );
  },
});

export const {
  selectAll: selectImagesAll,
  selectById: selectImagesById,
  selectEntities: selectImagesEntities,
  selectIds: selectImagesIds,
  selectTotal: selectImagesTotal,
} = imagesAdapter.getSelectors<RootState>((state) => state.gallery);

export const {
  imageUpserted,
  imageUpdatedOne,
  imageUpdatedMany,
  imageRemoved,
  imagesRemoved,
  imageCategoriesChanged,
  imageRangeEndSelected,
  imageSelectionToggled,
  imageSelected,
  shouldAutoSwitchChanged,
  setGalleryImageMinimumWidth,
  setGalleryView,
  boardIdSelected,
  isLoadingChanged,
} = gallerySlice.actions;

export default gallerySlice.reducer;

export const selectFilteredImagesLocal = createSelector(
  (state: typeof initialGalleryState) => state,
  (galleryState) => {
    const allImages = imagesAdapter.getSelectors().selectAll(galleryState);
    const { categories, selectedBoardId } = galleryState;

    const filteredImages = allImages.filter((i) => {
      const isInCategory = categories.includes(i.image_category);
      const isInSelectedBoard = selectedBoardId
        ? i.board_id === selectedBoardId
        : true;
      return isInCategory && isInSelectedBoard;
    });

    return filteredImages;
  }
);

export const selectFilteredImages = createSelector(
  (state: RootState) => state,
  (state) => {
    return selectFilteredImagesLocal(state.gallery);
  },
  defaultSelectorOptions
);

export const selectFilteredImagesAsObject = createSelector(
  selectFilteredImages,
  (filteredImages) => keyBy(filteredImages, 'image_name')
);

export const selectFilteredImagesIds = createSelector(
  selectFilteredImages,
  (filteredImages) => filteredImages.map((i) => i.image_name)
);

export const selectLastSelectedImage = createSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1],
  defaultSelectorOptions
);
