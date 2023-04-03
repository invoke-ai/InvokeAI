import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';

import * as InvokeAI from 'app/invokeai';
import { RootState } from 'app/store';

const resultsAdapter = createEntityAdapter<InvokeAI.ResultImage>({
  // Image IDs are just their filename
  selectId: (image) => image.name,
  // Keep the "all IDs" array sorted based on result timestamps
  sortComparer: (a, b) => b.timestamp - a.timestamp,
});

const resultsSlice = createSlice({
  name: 'results',
  initialState: resultsAdapter.getInitialState(),
  reducers: {
    // Can pass adapter functions directly as case reducers.  Because we're passing this
    // as a value, `createSlice` will auto-generate the action type / creator
    resultAdded: resultsAdapter.addOne,
    resultsReceived: resultsAdapter.setAll,
  },
});

// Can create a set of memoized selectors based on the location of this entity state
export const {
  selectAll: selectResultsAll,
  selectById: selectResultsById,
  selectEntities: selectResultsEntities,
  selectIds: selectResultsIds,
  selectTotal: selectResultsTotal,
} = resultsAdapter.getSelectors<RootState>((state) => state.results);

export const { resultAdded, resultsReceived } = resultsSlice.actions;

export default resultsSlice.reducer;
