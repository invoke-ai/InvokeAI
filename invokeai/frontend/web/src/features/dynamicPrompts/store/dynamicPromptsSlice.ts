import { PayloadAction, createSlice } from '@reduxjs/toolkit';

export interface DynamicPromptsState {
  isEnabled: boolean;
  maxPrompts: number;
  combinatorial: boolean;
}

export const initialDynamicPromptsState: DynamicPromptsState = {
  isEnabled: false,
  maxPrompts: 100,
  combinatorial: true,
};

const initialState: DynamicPromptsState = initialDynamicPromptsState;

export const dynamicPromptsSlice = createSlice({
  name: 'dynamicPrompts',
  initialState,
  reducers: {
    maxPromptsChanged: (state, action: PayloadAction<number>) => {
      state.maxPrompts = action.payload;
    },
    maxPromptsReset: (state) => {
      state.maxPrompts = initialDynamicPromptsState.maxPrompts;
    },
    combinatorialToggled: (state) => {
      state.combinatorial = !state.combinatorial;
    },
    isEnabledToggled: (state) => {
      state.isEnabled = !state.isEnabled;
    },
  },
});

export const {
  isEnabledToggled,
  maxPromptsChanged,
  maxPromptsReset,
  combinatorialToggled,
} = dynamicPromptsSlice.actions;

export default dynamicPromptsSlice.reducer;
