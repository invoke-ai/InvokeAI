import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  StableDiffusion1ModelCheckpointConfig,
  StableDiffusion1ModelDiffusersConfig,
} from 'services/api';

import { getModels } from 'services/thunks/model';

export type SD1ModelType = (
  | StableDiffusion1ModelCheckpointConfig
  | StableDiffusion1ModelDiffusersConfig
) & {
  name: string;
};

export const sd1ModelsAdapter = createEntityAdapter<SD1ModelType>({
  selectId: (model) => model.name,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

export const sd1InitialModelsState = sd1ModelsAdapter.getInitialState();

export type SD1ModelState = typeof sd1InitialModelsState;

export const sd1ModelsSlice = createSlice({
  name: 'sd1models',
  initialState: sd1InitialModelsState,
  reducers: {
    modelAdded: sd1ModelsAdapter.upsertOne,
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(getModels.fulfilled, (state, action) => {
      if (action.meta.arg.baseModel !== 'sd-1') return;
      sd1ModelsAdapter.setAll(state, action.payload);
    });
  },
});

export const {
  selectAll: selectAllSD1Models,
  selectById: selectByIdSD1Models,
  selectEntities: selectEntitiesSD1Models,
  selectIds: selectIdsSD1Models,
  selectTotal: selectTotalSD1Models,
} = sd1ModelsAdapter.getSelectors<RootState>((state) => state.sd1models);

export const { modelAdded } = sd1ModelsSlice.actions;

export default sd1ModelsSlice.reducer;
