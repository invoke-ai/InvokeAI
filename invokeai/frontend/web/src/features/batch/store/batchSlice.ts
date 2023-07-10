import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { uniq } from 'lodash-es';
import { imageDeleted } from 'services/api/thunks/image';

type BatchState = {
  isEnabled: boolean;
  imageNames: string[];
  asInitialImage: boolean;
  controlNets: string[];
  selection: string[];
};

export const initialBatchState: BatchState = {
  isEnabled: false,
  imageNames: [],
  asInitialImage: false,
  controlNets: [],
  selection: [],
};

const batch = createSlice({
  name: 'batch',
  initialState: initialBatchState,
  reducers: {
    isEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isEnabled = action.payload;
    },
    imageAddedToBatch: (state, action: PayloadAction<string>) => {
      state.imageNames.push(action.payload);
    },
    imagesAddedToBatch: (state, action: PayloadAction<string[]>) => {
      state.imageNames = state.imageNames.concat(action.payload);
    },
    imageRemovedFromBatch: (state, action: PayloadAction<string>) => {
      state.imageNames = state.imageNames.filter(
        (imageName) => action.payload !== imageName
      );
      state.selection = state.selection.filter(
        (imageName) => action.payload !== imageName
      );
    },
    imagesRemovedFromBatch: (state, action: PayloadAction<string[]>) => {
      state.imageNames = state.imageNames.filter(
        (imageName) => !action.payload.includes(imageName)
      );
      state.selection = state.selection.filter(
        (imageName) => !action.payload.includes(imageName)
      );
    },
    batchImageRangeEndSelected: (state, action: PayloadAction<string>) => {
      const rangeEndImageName = action.payload;
      const lastSelectedImage = state.selection[state.selection.length - 1];

      const { imageNames } = state;

      const lastClickedIndex = imageNames.findIndex(
        (n) => n === lastSelectedImage
      );
      const currentClickedIndex = imageNames.findIndex(
        (n) => n === rangeEndImageName
      );
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);

        const imagesToSelect = imageNames.slice(start, end + 1);

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
        : [String(state.imageNames[0])];
    },
    batchReset: (state) => {
      state.imageNames = [];
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
      state.imageNames = state.imageNames.filter(
        (imageName) => imageName !== action.meta.arg.image_name
      );
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
