import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { newSessionRequested } from 'features/controlLayers/store/actions';
import type { ParameterHRFMethod, ParameterStrength } from 'features/parameters/types/parameterSchemas';

interface HRFState {
  _version: 1;
  hrfEnabled: boolean;
  hrfStrength: ParameterStrength;
  hrfMethod: ParameterHRFMethod;
}

const initialHRFState: HRFState = {
  _version: 1,
  hrfStrength: 0.45,
  hrfEnabled: false,
  hrfMethod: 'ESRGAN',
};

export const hrfSlice = createSlice({
  name: 'hrf',
  initialState: initialHRFState,
  reducers: {
    setHrfStrength: (state, action: PayloadAction<number>) => {
      state.hrfStrength = action.payload;
    },
    setHrfEnabled: (state, action: PayloadAction<boolean>) => {
      state.hrfEnabled = action.payload;
    },
    setHrfMethod: (state, action: PayloadAction<ParameterHRFMethod>) => {
      state.hrfMethod = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addMatcher(newSessionRequested, () => {
      return deepClone(initialHRFState);
    });
  },
});

export const { setHrfEnabled, setHrfStrength, setHrfMethod } = hrfSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateHRFState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const hrfPersistConfig: PersistConfig<HRFState> = {
  name: hrfSlice.name,
  initialState: initialHRFState,
  migrate: migrateHRFState,
  persistDenylist: [],
};

export const selectHrfSlice = (state: RootState) => state.hrf;
export const selectHrfEnabled = createSelector(selectHrfSlice, (hrf) => hrf.hrfEnabled);
export const selectHrfMethod = createSelector(selectHrfSlice, (hrf) => hrf.hrfMethod);
export const selectHrfStrength = createSelector(selectHrfSlice, (hrf) => hrf.hrfStrength);
