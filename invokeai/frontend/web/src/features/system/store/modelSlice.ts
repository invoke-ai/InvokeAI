import { createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { CkptModelInfo, DiffusersModelInfo } from 'services/api';
import { receivedModels } from 'services/thunks/model';

export type Model = (CkptModelInfo | DiffusersModelInfo) & {
  name: string;
};

export const modelsAdapter = createEntityAdapter<Model>({
  selectId: (model) => model.name,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

export const initialModelsState = modelsAdapter.getInitialState();

export type ModelsState = typeof initialModelsState;

export const modelsSlice = createSlice({
  name: 'models',
  initialState: initialModelsState,
  reducers: {
    modelAdded: modelsAdapter.upsertOne,
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(receivedModels.fulfilled, (state, action) => {
      const models = action.payload;
      modelsAdapter.setAll(state, models);
    });
  },
});

export const {
  selectAll: selectModelsAll,
  selectById: selectModelsById,
  selectEntities: selectModelsEntities,
  selectIds: selectModelsIds,
  selectTotal: selectModelsTotal,
} = modelsAdapter.getSelectors<RootState>((state) => state.models);

export const { modelAdded } = modelsSlice.actions;

export default modelsSlice.reducer;
