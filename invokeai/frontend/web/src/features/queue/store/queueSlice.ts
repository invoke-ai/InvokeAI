import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import z from 'zod';

const zQueueState = z.object({
  listCursor: z.number().optional(),
  listPriority: z.number().optional(),
  selectedQueueItem: z.string().optional(),
  resumeProcessorOnEnqueue: z.boolean(),
});
type QueueState = z.infer<typeof zQueueState>;

const getInitialState = (): QueueState => ({
  listCursor: undefined,
  listPriority: undefined,
  selectedQueueItem: undefined,
  resumeProcessorOnEnqueue: true,
});

const slice = createSlice({
  name: 'queue',
  initialState: getInitialState(),
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

export const { listCursorChanged, listPriorityChanged, listParamsReset } = slice.actions;

export const queueSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zQueueState,
  getInitialState,
};

const selectQueueSlice = (state: RootState) => state.queue;
const createQueueSelector = <T>(selector: Selector<QueueState, T>) => createSelector(selectQueueSlice, selector);
export const selectQueueListCursor = createQueueSelector((queue) => queue.listCursor);
export const selectQueueListPriority = createQueueSelector((queue) => queue.listPriority);
