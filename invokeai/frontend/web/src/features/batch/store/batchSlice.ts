import {
  PayloadAction,
  createEntityAdapter,
  createSlice,
} from '@reduxjs/toolkit';
import { dateComparator } from 'common/util/dateComparator';
import { uniq } from 'lodash-es';
import { imageDeleted } from 'services/api/thunks/image';
import { ImageDTO } from 'services/api/types';

export const batchImagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.updated_at, a.updated_at),
});

type AdditionalBatchState = {
  isEnabled: boolean;
  asInitialImage: boolean;
  controlNets: string[];
  selection: string[];
};

export const initialBatchState =
  batchImagesAdapter.getInitialState<AdditionalBatchState>({
    isEnabled: false,
    asInitialImage: false,
    controlNets: [],
    selection: [],
  });

const batch = createSlice({
  name: 'batch',
  initialState: initialBatchState,
  reducers: {
    isEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isEnabled = action.payload;
    },
    imageAddedToBatch: (state, action: PayloadAction<ImageDTO>) => {
      batchImagesAdapter.addOne(state, action.payload);
    },
    imagesAddedToBatch: (state, action: PayloadAction<ImageDTO[]>) => {
      batchImagesAdapter.addMany(state, action.payload);
    },
    imageRemovedFromBatch: (state, action: PayloadAction<string>) => {
      batchImagesAdapter.removeOne(state, action.payload);
      state.selection = state.selection.filter(
        (imageName) => action.payload !== imageName
      );
    },
    imagesRemovedFromBatch: (state, action: PayloadAction<string[]>) => {
      batchImagesAdapter.removeMany(state, action.payload);
      state.selection = state.selection.filter(
        (imageName) => !action.payload.includes(imageName)
      );
    },
    batchImageRangeEndSelected: (state, action: PayloadAction<string>) => {
      const rangeEndImageName = action.payload;
      const lastSelectedImage = state.selection[state.selection.length - 1];

      const images = batchImagesAdapter.getSelectors().selectAll(state);
      const lastClickedIndex = images.findIndex(
        (n) => n.image_name === lastSelectedImage
      );
      const currentClickedIndex = images.findIndex(
        (n) => n.image_name === rangeEndImageName
      );
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);

        const imagesToSelect = images
          .slice(start, end + 1)
          .map((i) => i.image_name);

        state.selection = uniq(state.selection.concat(imagesToSelect));
      }
    },
    batchImageSelectionToggled: (state, action: PayloadAction<string>) => {
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
    batchImageSelected: (state, action: PayloadAction<string | null>) => {
      state.selection = action.payload
        ? [action.payload]
        : [String(state.ids[0])];
    },
    batchReset: (state) => {
      batchImagesAdapter.removeAll(state);
      state.selection = [];
    },
    asInitialImageToggled: (state) => {
      state.asInitialImage = !state.asInitialImage;
    },
    controlNetAddedToBatch: (state, action: PayloadAction<string>) => {
      state.controlNets = uniq(state.controlNets.concat(action.payload));
    },
    controlNetRemovedFromBatch: (state, action: PayloadAction<string>) => {
      state.controlNets = state.controlNets.filter(
        (controlNetId) => controlNetId !== action.payload
      );
    },
    controlNetToggled: (state, action: PayloadAction<string>) => {
      if (state.controlNets.includes(action.payload)) {
        state.controlNets = state.controlNets.filter(
          (controlNetId) => controlNetId !== action.payload
        );
      } else {
        state.controlNets = uniq(state.controlNets.concat(action.payload));
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(imageDeleted.fulfilled, (state, action) => {
      batchImagesAdapter.removeOne(state, action.meta.arg.image_name);
      state.selection = state.selection.filter(
        (imageName) => imageName !== action.meta.arg.image_name
      );
    });
  },
});

export const {
  isEnabledChanged,
  imageAddedToBatch,
  imagesAddedToBatch,
  imageRemovedFromBatch,
  imagesRemovedFromBatch,
  asInitialImageToggled,
  controlNetAddedToBatch,
  controlNetRemovedFromBatch,
  batchReset,
  controlNetToggled,
  batchImageRangeEndSelected,
  batchImageSelectionToggled,
  batchImageSelected,
} = batch.actions;

export default batch.reducer;
