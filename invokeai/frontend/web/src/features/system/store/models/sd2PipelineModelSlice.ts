import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  StableDiffusion2ModelCheckpointConfig,
  StableDiffusion2ModelDiffusersConfig,
} from 'services/api';

import { receivedModels } from 'services/thunks/model';

export type SD2PipelineModelType = (
  | StableDiffusion2ModelCheckpointConfig
  | StableDiffusion2ModelDiffusersConfig
) & {
  name: string;
};

export const sd2PipelineModelsAdapater =
  createEntityAdapter<SD2PipelineModelType>({
    selectId: (model) => model.name,
    sortComparer: (a, b) => a.name.localeCompare(b.name),
  });

export const sd2InitialPipelineModelsState =
  sd2PipelineModelsAdapater.getInitialState();

export type SD2PipelineModelState = typeof sd2InitialPipelineModelsState;

export const sd2PipelineModelsSlice = createSlice({
  name: 'sd2PipelineModels',
  initialState: sd2InitialPipelineModelsState,
  reducers: {
    modelAdded: sd2PipelineModelsAdapater.upsertOne,
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(receivedModels.fulfilled, (state, action) => {
      if (action.meta.arg.baseModel !== 'sd-2') return;
      sd2PipelineModelsAdapater.setAll(state, action.payload);
    });
  },
});

export const {
  selectAll: selectAllSD2PipelineModels,
  selectById: selectByIdSD2PipelineModels,
  selectEntities: selectEntitiesSD2PipelineModels,
  selectIds: selectIdsSD2PipelineModels,
  selectTotal: selectTotalSD2PipelineModels,
} = sd2PipelineModelsAdapater.getSelectors<RootState>(
  (state) => state.sd2pipelinemodels
);

export const { modelAdded } = sd2PipelineModelsSlice.actions;

export default sd2PipelineModelsSlice.reducer;
