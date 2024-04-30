import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { ParameterScheduler, ParameterSDXLRefinerModel } from 'features/parameters/types/parameterSchemas';

type SDXLState = {
  _version: 2;
  refinerModel: ParameterSDXLRefinerModel | null;
  refinerSteps: number;
  refinerCFGScale: number;
  refinerScheduler: ParameterScheduler;
  refinerPositiveAestheticScore: number;
  refinerNegativeAestheticScore: number;
  refinerStart: number;
};

const initialSDXLState: SDXLState = {
  _version: 2,
  refinerModel: null,
  refinerSteps: 20,
  refinerCFGScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
};

export const sdxlSlice = createSlice({
  name: 'sdxl',
  initialState: initialSDXLState,
  reducers: {
    refinerModelChanged: (state, action: PayloadAction<ParameterSDXLRefinerModel | null>) => {
      state.refinerModel = action.payload;
    },
    setRefinerSteps: (state, action: PayloadAction<number>) => {
      state.refinerSteps = action.payload;
    },
    setRefinerCFGScale: (state, action: PayloadAction<number>) => {
      state.refinerCFGScale = action.payload;
    },
    setRefinerScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.refinerScheduler = action.payload;
    },
    setRefinerPositiveAestheticScore: (state, action: PayloadAction<number>) => {
      state.refinerPositiveAestheticScore = action.payload;
    },
    setRefinerNegativeAestheticScore: (state, action: PayloadAction<number>) => {
      state.refinerNegativeAestheticScore = action.payload;
    },
    setRefinerStart: (state, action: PayloadAction<number>) => {
      state.refinerStart = action.payload;
    },
  },
});

export const {
  refinerModelChanged,
  setRefinerSteps,
  setRefinerCFGScale,
  setRefinerScheduler,
  setRefinerPositiveAestheticScore,
  setRefinerNegativeAestheticScore,
  setRefinerStart,
} = sdxlSlice.actions;

export const selectSdxlSlice = (state: RootState) => state.sdxl;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateSDXLState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  if (state._version === 1) {
    // Model type has changed, so we need to reset the state - too risky to migrate
    state._version = 2;
    state.refinerModel = null;
  }
  return state;
};

export const sdxlPersistConfig: PersistConfig<SDXLState> = {
  name: sdxlSlice.name,
  initialState: initialSDXLState,
  migrate: migrateSDXLState,
  persistDenylist: [],
};
