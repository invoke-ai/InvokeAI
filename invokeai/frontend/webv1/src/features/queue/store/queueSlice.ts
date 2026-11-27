import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { type GetQueueItemIdsArgs, zSQLiteDirection } from 'services/api/types';
import z from 'zod';

const zQueueState = z.object({
  sortOrder: zSQLiteDirection,
});
type QueueState = z.infer<typeof zQueueState>;

const getInitialState = (): QueueState => ({
  sortOrder: 'DESC',
});

const slice = createSlice({
  name: 'queue',
  initialState: getInitialState(),
  reducers: {
    sortOrderChanged: (state, action: PayloadAction<'ASC' | 'DESC'>) => {
      state.sortOrder = action.payload;
    },
  },
});

export const { sortOrderChanged } = slice.actions;

export const queueSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zQueueState,
  getInitialState,
};

const selectQueueSlice = (state: RootState) => state.queue;
const createQueueSelector = <T>(selector: Selector<QueueState, T>) => createSelector(selectQueueSlice, selector);
export const selectQueueSortOrder = createQueueSelector((queue) => queue.sortOrder);
export const selectGetQueueItemIdsArgs = createMemoizedSelector(
  [selectQueueSortOrder],
  (order_dir): GetQueueItemIdsArgs => ({
    order_dir,
  })
);
