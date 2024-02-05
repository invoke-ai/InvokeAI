import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';

export interface QueueState {
  listCursor: number | undefined;
  listPriority: number | undefined;
  selectedQueueItem: string | undefined;
  resumeProcessorOnEnqueue: boolean;
}

export const initialQueueState: QueueState = {
  listCursor: undefined,
  listPriority: undefined,
  selectedQueueItem: undefined,
  resumeProcessorOnEnqueue: true,
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
    queueItemSelectionToggled: (state, action: PayloadAction<string | undefined>) => {
      if (state.selectedQueueItem === action.payload) {
        state.selectedQueueItem = undefined;
      } else {
        state.selectedQueueItem = action.payload;
      }
    },
    resumeProcessorOnEnqueueChanged: (state, action: PayloadAction<boolean>) => {
      state.resumeProcessorOnEnqueue = action.payload;
    },
  },
});

export const {
  listCursorChanged,
  listPriorityChanged,
  listParamsReset,
  queueItemSelectionToggled,
  resumeProcessorOnEnqueueChanged,
} = queueSlice.actions;

export const selectQueueSlice = (state: RootState) => state.queue;
