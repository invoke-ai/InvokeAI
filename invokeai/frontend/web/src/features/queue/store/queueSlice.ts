import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { type GetQueueItemIdsArgs, zSQLiteDirection } from 'services/api/types';
import z from 'zod';

const zSortBy = z.enum(['status', 'created_at', 'completed_at']);
export type SortBy = z.infer<typeof zSortBy>;

const zQueueState = z.object({
  listCursor: z.number().optional(),
  listPriority: z.number().optional(),
  selectedQueueItem: z.string().optional(),
  sortBy: zSortBy,
  sortOrder: zSQLiteDirection,
  resumeProcessorOnEnqueue: z.boolean(),
});
type QueueState = z.infer<typeof zQueueState>;

const getInitialState = (): QueueState => ({
  listCursor: undefined,
  listPriority: undefined,
  selectedQueueItem: undefined,
  sortBy: 'created_at',
  sortOrder: 'DESC',
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
    sortByChanged: (state, action: PayloadAction<SortBy>) => {
      state.sortBy = action.payload;
    },
    sortOrderChanged: (state, action: PayloadAction<'ASC' | 'DESC'>) => {
      state.sortOrder = action.payload;
    },
  },
});

export const { listCursorChanged, listPriorityChanged, listParamsReset, sortByChanged, sortOrderChanged } =
  slice.actions;

export const queueSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zQueueState,
  getInitialState,
};

const selectQueueSlice = (state: RootState) => state.queue;
const createQueueSelector = <T>(selector: Selector<QueueState, T>) => createSelector(selectQueueSlice, selector);
export const selectQueueSortBy = createQueueSelector((queue) => queue.sortBy);
export const selectQueueSortOrder = createQueueSelector((queue) => queue.sortOrder);
export const selectGetQueueItemIdsArgs = createMemoizedSelector(
  [selectQueueSortBy, selectQueueSortOrder],
  (order_by, order_dir): GetQueueItemIdsArgs => ({
    order_by,
    order_dir,
  })
);
