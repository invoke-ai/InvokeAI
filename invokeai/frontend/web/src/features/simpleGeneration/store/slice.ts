import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { SimpleGenerationState } from 'features/simpleGeneration/store/types';
import { zSimpleGenerationState } from 'features/simpleGeneration/store/types';

const getInitialState = (): SimpleGenerationState => zSimpleGenerationState.parse({});

export const simpleGenerationSlice = createSlice({
  name: 'simpleGeneration',
  initialState: getInitialState(),
  reducers: {
    positivePromptChanged: (
      state,
      action: PayloadAction<{
        positivePrompt: SimpleGenerationState['positivePrompt'];
      }>
    ) => {
      const { positivePrompt } = action.payload;
      state.positivePrompt = positivePrompt;
    },
    modelChanged: (
      state,
      action: PayloadAction<{
        model: SimpleGenerationState['model'];
      }>
    ) => {
      const { model } = action.payload;
      state.model = model;
    },
    aspectRatioChanged: (
      state,
      action: PayloadAction<{
        aspectRatio: SimpleGenerationState['aspectRatio'];
      }>
    ) => {
      const { aspectRatio } = action.payload;
      state.aspectRatio = aspectRatio;
    },
    startingImageChanged: (
      state,
      action: PayloadAction<{
        startingImage: SimpleGenerationState['startingImage'];
      }>
    ) => {
      const { startingImage } = action.payload;
      state.startingImage = startingImage;
    },
    referenceImageChanged: (
      state,
      action: PayloadAction<{
        index: number;
        referenceImage: SimpleGenerationState['referenceImages'][number];
      }>
    ) => {
      const { index, referenceImage } = action.payload;
      state.referenceImages[index] = referenceImage;
    },
    controlImageChanged: (
      state,
      action: PayloadAction<{
        controlImage: SimpleGenerationState['controlImage'];
      }>
    ) => {
      const { controlImage } = action.payload;
      state.controlImage = controlImage;
    },
    reset: () => getInitialState(),
  },
});

export const {
  aspectRatioChanged,
  controlImageChanged,
  modelChanged,
  positivePromptChanged,
  referenceImageChanged,
  startingImageChanged,
  reset,
} = simpleGenerationSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateSimpleGenerationState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const simpleGenerationPersistConfig: PersistConfig<SimpleGenerationState> = {
  name: simpleGenerationSlice.name,
  initialState: getInitialState(),
  migrate: migrateSimpleGenerationState,
  persistDenylist: [],
};

export const selectSimpleGenerationSlice = (state: RootState) => state.simpleGeneration;
const createSliceSelector = <T>(selector: Selector<SimpleGenerationState, T>) =>
  createSelector(selectSimpleGenerationSlice, selector);

export const selectPositivePrompt = createSliceSelector((slice) => slice.positivePrompt);
export const selectModel = createSliceSelector((slice) => slice.model);
// export const selectModelBase = createSliceSelector((slice) => slice.model?.base);
// export const selectModelKey = createSliceSelector((slice) => slice.model?.key);
export const selectAspectRatio = createSliceSelector((slice) => slice.aspectRatio);
