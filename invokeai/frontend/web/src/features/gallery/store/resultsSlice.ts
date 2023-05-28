import {
  PayloadAction,
  createEntityAdapter,
  createSlice,
} from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import {
  receivedResultImagesPage,
  IMAGES_PER_PAGE,
} from 'services/thunks/gallery';
import { ImageDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';

export type ResultsImageDTO = Omit<ImageDTO, 'image_type'> & {
  image_type: 'results';
};

export const resultsAdapter = createEntityAdapter<ResultsImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.created_at, a.created_at),
});

type AdditionalResultsState = {
  page: number;
  pages: number;
  isLoading: boolean;
  nextPage: number;
  upsertedImageCount: number;
};

export const initialResultsState =
  resultsAdapter.getInitialState<AdditionalResultsState>({
    page: 0,
    pages: 0,
    isLoading: false,
    nextPage: 0,
    upsertedImageCount: 0,
  });

export type ResultsState = typeof initialResultsState;

const resultsSlice = createSlice({
  name: 'results',
  initialState: initialResultsState,
  reducers: {
    resultUpserted: (state, action: PayloadAction<ResultsImageDTO>) => {
      resultsAdapter.upsertOne(state, action.payload);
      state.upsertedImageCount += 1;
    },
  },
  extraReducers: (builder) => {
    /**
     * Received Result Images Page - PENDING
     */
    builder.addCase(receivedResultImagesPage.pending, (state) => {
      state.isLoading = true;
    });

    /**
     * Received Result Images Page - FULFILLED
     */
    builder.addCase(receivedResultImagesPage.fulfilled, (state, action) => {
      const { page, pages } = action.payload;

      // We know these will all be of the results type, but it's not represented in the API types
      const items = action.payload.items as ResultsImageDTO[];

      resultsAdapter.setMany(state, items);

      state.page = page;
      state.pages = pages;
      state.nextPage = items.length < IMAGES_PER_PAGE ? page : page + 1;
      state.isLoading = false;
    });
  },
});

export const {
  selectAll: selectResultsAll,
  selectById: selectResultsById,
  selectEntities: selectResultsEntities,
  selectIds: selectResultsIds,
  selectTotal: selectResultsTotal,
} = resultsAdapter.getSelectors<RootState>((state) => state.results);

export const { resultUpserted } = resultsSlice.actions;

export default resultsSlice.reducer;
