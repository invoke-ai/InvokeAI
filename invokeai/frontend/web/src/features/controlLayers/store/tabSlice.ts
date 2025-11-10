import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { SerializedStateFromDenyList, SliceConfig } from 'app/store/types';
import { extractTabActionContext } from 'app/store/util';
import { isPlainObject } from 'es-toolkit';
import { merge, omit } from 'es-toolkit/compat';
import { assert } from 'tsafe';

import { getInitialLoRAsState, isLoRAsStateAction, lorasSlice } from './lorasSlice';
import { getInitialParamsState as getInitialParamsState, isTabParamsStateAction, paramsState } from './paramsSlice';
import { getInitialRefImagesState, isRefImagesStateAction, refImagesDenyList, refImagesSlice } from './refImagesSlice';
import type { RefImagesState, TabInstanceState as TabInstanceParamsState, TabName, TabState } from './types';
import { zTabState } from './types';

export const getInitialTabInstanceParamsState = (): TabInstanceParamsState => ({
  loras: getInitialLoRAsState(),
  params: getInitialParamsState(),
  refImages: getInitialRefImagesState(),
});

const getInitialTabState = (): TabState => ({
  activeTab: 'canvas' as const,
  generate: getInitialTabInstanceParamsState(),
  upscaling: getInitialTabInstanceParamsState(),
});

const tabSlice = createSlice({
  name: 'tab',
  initialState: getInitialTabState(),
  reducers: {
    setActiveTab: (state, action: PayloadAction<TabName>) => {
      state.activeTab = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addMatcher(isTabInstanceParamsAction, (state, action) => {
      const context = extractTabActionContext(action);

      if (!context) {
        return;
      }

      switch (context.tab) {
        case 'generate':
          state.generate = tabInstanceParamsSlice.reducer(state.generate, action);
          break;
        case 'upscaling':
          state.upscaling = tabInstanceParamsSlice.reducer(state.upscaling, action);
          break;
      }
    });
  },
});

export const tabInstanceParamsSlice = createSlice({
  name: 'tabInstance',
  initialState: {} as TabInstanceParamsState,
  reducers: {},
  extraReducers(builder) {
    builder.addDefaultCase((state, action) => {
      if (isLoRAsStateAction(action)) {
        state.loras = lorasSlice.reducer(state.loras, action);
      }
      if (isTabParamsStateAction(action)) {
        state.params = paramsState.reducer(state.params, action);
      }
      if (isRefImagesStateAction(action)) {
        state.refImages = refImagesSlice.reducer(state.refImages, action);
      }
    });
  },
});

export const { setActiveTab } = tabSlice.actions;

export const isTabInstanceParamsAction = (action: UnknownAction) =>
  isLoRAsStateAction(action) || isTabParamsStateAction(action) || isRefImagesStateAction(action);

type SerializedRefImagesState = SerializedStateFromDenyList<RefImagesState, typeof refImagesDenyList>;
type SerializedTabInstanceParamsState = Omit<TabInstanceParamsState, 'refImages'> & {
  refImages: SerializedRefImagesState;
};
type SerializedTabState = Omit<TabState, 'generate' | 'upscaling'> & {
  generate: SerializedTabInstanceParamsState;
  upscaling: SerializedTabInstanceParamsState;
};

export const tabSliceConfig: SliceConfig<typeof tabSlice, TabState, SerializedTabState> = {
  slice: tabSlice,
  getInitialState: getInitialTabState,
  schema: zTabState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      return zTabState.parse(state);
    },
    serialize: (state) => ({
      ...state,
      generate: {
        ...state.generate,
        refImages: omit(state.generate.refImages, refImagesDenyList),
      },
      upscaling: {
        ...state.upscaling,
        refImages: omit(state.upscaling.refImages, refImagesDenyList),
      },
    }),
    deserialize: (state) => {
      const tabInstanceParamsState = state as SerializedTabState;

      return merge(tabInstanceParamsState, getInitialTabState());
    },
  },
};
