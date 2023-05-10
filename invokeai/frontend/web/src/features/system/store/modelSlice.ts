import { createEntityAdapter, PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { keys, sample } from 'lodash-es';
import { CkptModelInfo, DiffusersModelInfo } from 'services/api';
import { receivedModels } from 'services/thunks/model';

export type Model = (CkptModelInfo | DiffusersModelInfo) & {
  name: string;
};

export const modelsAdapter = createEntityAdapter<Model>({
  selectId: (model) => model.name,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

type AdditionalModelsState = {
  selectedModelName: string;
};

export const initialModelsState =
  modelsAdapter.getInitialState<AdditionalModelsState>({
    selectedModelName: '',
  });

export type ModelsState = typeof initialModelsState;

export const modelsSlice = createSlice({
  name: 'models',
  initialState: initialModelsState,
  reducers: {
    modelAdded: modelsAdapter.upsertOne,
    modelSelected: (state, action: PayloadAction<string>) => {
      state.selectedModelName = action.payload;
    },
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(receivedModels.fulfilled, (state, action) => {
      const models = action.payload;
      modelsAdapter.setAll(state, models);

      // If the current selected model is `''` or isn't actually in the list of models,
      // choose a random model
      if (
        !state.selectedModelName ||
        !keys(models).includes(state.selectedModelName)
      ) {
        const randomModel = sample(models);

        if (randomModel) {
          state.selectedModelName = randomModel.name;
        } else {
          state.selectedModelName = '';
        }
      }
    });
  },
});

export const selectedModelSelector = (state: RootState) => {
  const { selectedModelName } = state.models;
  const selectedModel = selectModelsById(state, selectedModelName);

  return selectedModel ?? null;
};

export const {
  selectAll: selectModelsAll,
  selectById: selectModelsById,
  selectEntities: selectModelsEntities,
  selectIds: selectModelsIds,
  selectTotal: selectModelsTotal,
} = modelsAdapter.getSelectors<RootState>((state) => state.models);

export const { modelAdded, modelSelected } = modelsSlice.actions;

export default modelsSlice.reducer;
