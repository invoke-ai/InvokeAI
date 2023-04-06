import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { ModelsList } from 'services/api';
import { receivedModels } from 'services/thunks/model';

export interface ModelState {
  modelList: ModelsList['models'];
  currentModel?: string;
}

const initialModelState: ModelState = {
  modelList: {},
  currentModel: undefined,
};

export const modelSlice = createSlice({
  name: 'model',
  initialState: initialModelState,
  reducers: {
    setModelList: (state, action: PayloadAction<ModelsList['models']>) => {
      state.modelList = action.payload;
    },
    setCurrentModel: (state, action: PayloadAction<string>) => {
      state.currentModel = action.payload;
    },
  },
  extraReducers(builder) {
    /**
     * Received Models - FULFILLED
     */
    builder.addCase(receivedModels.fulfilled, (state, action) => {
      const models = action.payload.models;
      state.modelList = models;
    });
  },
});

export const { setModelList, setCurrentModel } = modelSlice.actions;

export default modelSlice.reducer;
