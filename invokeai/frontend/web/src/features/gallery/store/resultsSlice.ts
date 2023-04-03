import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { ResultImage } from 'app/invokeai';

import { RootState } from 'app/store';
import { map } from 'lodash';
import { getNextResultsPage } from 'services/thunks/extra';
import { isImageOutput } from 'services/types/guards';
import { prepareResultImage } from 'services/util/prepareResultImage';

// use `createEntityAdapter` to create a slice for results images
// https://redux-toolkit.js.org/api/createEntityAdapter#overview

// the "Entity" is InvokeAI.ResultImage, while the "entities" are instances of that type
const resultsAdapter = createEntityAdapter<ResultImage>({
  // Provide a callback to get a stable, unique identifier for each entity. This defaults to
  // `(item) => item.id`, but for our result images, the `name` is the unique identifier.
  selectId: (image) => image.name,
  // Order all images by their time (in descending order)
  sortComparer: (a, b) => b.timestamp - a.timestamp,
});

// This type is intersected with the Entity type to create the shape of the state
type AdditionalResultsState = {
  // these are a bit misleading; they refer to sessions, not results, but we don't have a route
  // to list all images directly at this time...
  page: number; // current page we are on
  pages: number; // the total number of pages available
  isLoading: boolean; // whether we are loading more images or not, mostly a placeholder
};

const resultsSlice = createSlice({
  name: 'results',
  initialState: resultsAdapter.getInitialState<AdditionalResultsState>({
    // provide the additional initial state
    page: 0,
    pages: 0,
    isLoading: false,
  }),
  reducers: {
    // the adapter provides some helper reducers; see the docs for all of them
    // can use them as helper functions within a reducer, or use the function itself as a reducer

    // here we just use the function itself as the reducer. we'll call this on `invocation_complete`
    // to add a single result
    resultAdded: resultsAdapter.addOne,
  },
  extraReducers: (builder) => {
    // here we can respond to a fulfilled call of the `getNextResultsPage` thunk
    // because we pass in the fulfilled thunk action creator, everything is typed
    builder.addCase(getNextResultsPage.pending, (state, action) => {
      state.isLoading = true;
    });
    builder.addCase(getNextResultsPage.fulfilled, (state, action) => {
      const { items, page, pages } = action.payload;

      // build flattened array of results ojects, use lodash `map()` to make results object an array
      const allResults = items.flatMap((session) => map(session.results));

      // filter out non-image-outputs (eg latents, prompts, etc)
      const imageOutputResults = allResults.filter(isImageOutput);

      // map results to ResultImage objects
      const resultImages = imageOutputResults.map((result) =>
        prepareResultImage(result.image)
      );

      // use the adapter reducer to add all the results to resultsSlice state
      resultsAdapter.addMany(state, resultImages);

      state.page = page;
      state.pages = pages;
      state.isLoading = false;
    });
  },
});

// Create a set of memoized selectors based on the location of this entity state
// to be used as selectors in a `useAppSelector()` call
export const {
  selectAll: selectResultsAll,
  selectById: selectResultsById,
  selectEntities: selectResultsEntities,
  selectIds: selectResultsIds,
  selectTotal: selectResultsTotal,
} = resultsAdapter.getSelectors<RootState>((state) => state.results);

export const { resultAdded } = resultsSlice.actions;

export default resultsSlice.reducer;
