import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

interface QueueState {
  listCursor: number | undefined;
  listPriority: number | undefined;
  selectedQueueItem: string | undefined;
  resumeProcessorOnEnqueue: boolean;
}

const initialQueueState: QueueState = {
  listCursor: undefined,
  listPriority: undefined,
  selectedQueueItem: undefined,
  resumeProcessorOnEnqueue: true,
};

export const queueSlice = createSlice({
  name: 'queue',
  initialState: initialQueueState,
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

export const { listCursorChanged, listPriorityChanged, listParamsReset } = queueSlice.actions;
