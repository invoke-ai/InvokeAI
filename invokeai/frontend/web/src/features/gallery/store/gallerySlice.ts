import type { PayloadAction, Update } from '@reduxjs/toolkit';
import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { dateComparator } from 'common/util/dateComparator';
import { uniq } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import { ImageCategory, ImageDTO } from 'services/api/types';
import { selectFilteredImagesLocal } from './gallerySelectors';

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

export type GalleryView = 'images' | 'assets';
export type BoardId =
  | 'all'
  | 'none'
  | 'batch'
  | (string & Record<never, never>);

type AdditionaGalleryState = {
  selection: string[];
  shouldAutoSwitch: boolean;
  galleryImageMinimumWidth: number;
  galleryView: GalleryView;
  selectedBoardId: BoardId;
  batchImageNames: string[];
  isBatchEnabled: boolean;
};

export const initialGalleryState =
  imagesAdapter.getInitialState<AdditionaGalleryState>({
    selection: [],
    shouldAutoSwitch: true,
    galleryImageMinimumWidth: 96,
    galleryView: 'images',
    selectedBoardId: 'all',
    batchImageNames: [],
    isBatchEnabled: false,
  });

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageUpserted: (state, action: PayloadAction<ImageDTO>) => {
      // TODO: port all instances of this to use RTK Query cache
      // imagesAdapter.upsertOne(state, action.payload);
      // if (
      //   state.shouldAutoSwitch &&
      //   action.payload.image_category === 'general'
      // ) {
      //   state.selection = [action.payload.image_name];
      //   state.galleryView = 'images';
      //   state.selectedBoardId = 'all';
      // }
    },
    imageUpdatedOne: (state, action: PayloadAction<Update<ImageDTO>>) => {
      // TODO: port all instances of this to use RTK Query cache
      // imagesAdapter.updateOne(state, action.payload);
    },
    imageRemoved: (state, action: PayloadAction<string>) => {
      // TODO: port all instances of this to use RTK Query cache
      // imagesAdapter.removeOne(state, action.payload);
      // state.batchImageNames = state.batchImageNames.filter(
      //   (name) => name !== action.payload
      // );
    },
    imagesRemoved: (state, action: PayloadAction<string[]>) => {
      // TODO: port all instances of this to use RTK Query cache
      // imagesAdapter.removeMany(state, action.payload);
      // state.batchImageNames = state.batchImageNames.filter(
      //   (name) => !action.payload.includes(name)
      // );
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
      state.selection = action.payload ? [action.payload] : [];
    },
    shouldAutoSwitchChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoSwitch = action.payload;
    },
    setGalleryImageMinimumWidth: (state, action: PayloadAction<number>) => {
      state.galleryImageMinimumWidth = action.payload;
    },
    setGalleryView: (state, action: PayloadAction<GalleryView>) => {
      state.galleryView = action.payload;
    },
    boardIdSelected: (state, action: PayloadAction<BoardId>) => {
      state.selectedBoardId = action.payload;
    },
    isBatchEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isBatchEnabled = action.payload;
    },
    imagesAddedToBatch: (state, action: PayloadAction<string[]>) => {
      state.batchImageNames = uniq(
        state.batchImageNames.concat(action.payload)
      );
    },
    imagesRemovedFromBatch: (state, action: PayloadAction<string[]>) => {
      state.batchImageNames = state.batchImageNames.filter(
        (imageName) => !action.payload.includes(imageName)
      );

      const newSelection = state.selection.filter(
        (imageName) => !action.payload.includes(imageName)
      );

      if (newSelection.length) {
        state.selection = newSelection;
        return;
      }

      state.selection = [state.batchImageNames[0]] ?? [];
    },
    batchReset: (state) => {
      state.batchImageNames = [];
      state.selection = [];
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      boardsApi.endpoints.deleteBoard.matchFulfilled,
      (state, action) => {
        if (action.meta.arg.originalArgs === state.selectedBoardId) {
          state.selectedBoardId = 'all';
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
  imageRemoved,
  imagesRemoved,
  imageRangeEndSelected,
  imageSelectionToggled,
  imageSelected,
  shouldAutoSwitchChanged,
  setGalleryImageMinimumWidth,
  setGalleryView,
  boardIdSelected,
  isBatchEnabledChanged,
  imagesAddedToBatch,
  imagesRemovedFromBatch,
} = gallerySlice.actions;

export default gallerySlice.reducer;
