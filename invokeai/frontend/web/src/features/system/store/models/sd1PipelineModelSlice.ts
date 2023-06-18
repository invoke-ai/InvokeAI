import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  StableDiffusion1ModelCheckpointConfig,
  StableDiffusion1ModelDiffusersConfig,
} from 'services/api';

import { getModels } from 'services/thunks/model';

export type SD1PipelineModelType = (
  | StableDiffusion1ModelCheckpointConfig
  | StableDiffusion1ModelDiffusersConfig
) & {
  name: string;
};

export const sd1PipelineModelsAdapter =
  createEntityAdapter<SD1PipelineModelType>({
    selectId: (model) => model.name,
    sortComparer: (a, b) => a.name.localeCompare(b.name),
  });

export const sd1InitialPipelineModelsState =
  sd1PipelineModelsAdapter.getInitialState();

export type SD1PipelineModelState = typeof sd1InitialPipelineModelsState;

export const sd1PipelineModelsSlice = createSlice({
  name: 'sd1models',
  initialState: sd1InitialPipelineModelsState,
  reducers: {
    modelAdded: sd1PipelineModelsAdapter.upsertOne,
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(getModels.fulfilled, (state, action) => {
      if (action.meta.arg.baseModel !== 'sd-1') return;
      sd1PipelineModelsAdapter.setAll(state, action.payload);
    });
  },
});

export const {
  selectAll: selectAllSD1PipelineModels,
  selectById: selectByIdSD1PipelineModels,
  selectEntities: selectEntitiesSD1PipelineModels,
  selectIds: selectIdsSD1PipelineModels,
  selectTotal: selectTotalSD1PipelineModels,
} = sd1PipelineModelsAdapter.getSelectors<RootState>(
  (state) => state.sd1pipelinemodels
);

export const { modelAdded } = sd1PipelineModelsSlice.actions;

export default sd1PipelineModelsSlice.reducer;
