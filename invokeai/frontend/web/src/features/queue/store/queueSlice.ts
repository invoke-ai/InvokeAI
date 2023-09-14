import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { queueApi } from 'services/api/endpoints/queue';

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
  },
  extraReducers(builder) {
    builder.addMatcher(
      queueApi.endpoints.enqueueBatch.matchPending,
      (state) => {
        state.listCursor = undefined;
        state.listPriority = undefined;
      }
    );
    builder.addMatcher(
      queueApi.endpoints.enqueueGraph.matchPending,
      (state) => {
        state.listCursor = undefined;
        state.listPriority = undefined;
      }
    );
  },
});

export const { listCursorChanged, listPriorityChanged } = queueSlice.actions;

export default queueSlice.reducer;
