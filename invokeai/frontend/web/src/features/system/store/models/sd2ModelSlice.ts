import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  StableDiffusion2ModelCheckpointConfig,
  StableDiffusion2ModelDiffusersConfig,
} from 'services/api';

import { getModels } from 'services/thunks/model';

export type SD2ModelType = (
  | StableDiffusion2ModelCheckpointConfig
  | StableDiffusion2ModelDiffusersConfig
) & {
  name: string;
};

export const sd2ModelsAdapater = createEntityAdapter<SD2ModelType>({
  selectId: (model) => model.name,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

export const sd2InitialModelsState = sd2ModelsAdapater.getInitialState();

export type SD2ModelState = typeof sd2InitialModelsState;

export const sd2ModelsSlice = createSlice({
  name: 'sd2models',
  initialState: sd2InitialModelsState,
  reducers: {
    modelAdded: sd2ModelsAdapater.upsertOne,
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(getModels.fulfilled, (state, action) => {
      if (action.meta.arg.baseModel !== 'sd-2') return;
      sd2ModelsAdapater.setAll(state, action.payload);
    });
  },
});

export const {
  selectAll: selectAllSD2Models,
  selectById: selectByIdSD2Models,
  selectEntities: selectEntitiesSD2Models,
  selectIds: selectIdsSD2Models,
  selectTotal: selectTotalSD2Models,
} = sd2ModelsAdapater.getSelectors<RootState>((state) => state.sd2models);

export const { modelAdded } = sd2ModelsSlice.actions;

export default sd2ModelsSlice.reducer;
