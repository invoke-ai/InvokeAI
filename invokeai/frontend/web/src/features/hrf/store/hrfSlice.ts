import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type {
  ParameterHRFMethod,
  ParameterStrength,
} from 'features/parameters/types/parameterSchemas';

export interface HRFState {
  hrfEnabled: boolean;
  hrfStrength: ParameterStrength;
  hrfMethod: ParameterHRFMethod;
}

export const initialHRFState: HRFState = {
  hrfStrength: 0.45,
  hrfEnabled: false,
  hrfMethod: 'ESRGAN',
};

const initialState: HRFState = initialHRFState;

export const hrfSlice = createSlice({
  name: 'hrf',
  initialState,
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
});

export const { setHrfEnabled, setHrfStrength, setHrfMethod } = hrfSlice.actions;

export default hrfSlice.reducer;
