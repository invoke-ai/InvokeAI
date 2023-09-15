import { PayloadAction, createSlice } from '@reduxjs/toolkit';

export interface QueueState {
  listCursor: number | undefined;
  listPriority: number | undefined;
}

export const initialQueueState: QueueState = {
  listCursor: undefined,
  listPriority: undefined,
};

const initialState: QueueState = initialQueueState;

export const queueSlice = createSlice({
  name: 'queue',
  initialState,
  reducers: {
    listCursorChanged: (state, action: PayloadAction<number | undefined>) => {
      state.listCursor = action.payload;
    },
    listPriorityChanged: (state, action: PayloadAction<number | undefined>) => {
      state.listPriority = action.payload;
    },
    listParamsReset: (state) => {
      state.listCursor = undefined;
      state.listPriority = undefined;
    },
  },
});

export const { listCursorChanged, listPriorityChanged, listParamsReset } =
  queueSlice.actions;

export default queueSlice.reducer;
