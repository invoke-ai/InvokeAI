import { PayloadAction, createSlice } from '@reduxjs/toolkit';

export type SeedBehaviour = 'PER_ITERATION' | 'PER_PROMPT';
export interface DynamicPromptsState {
  maxPrompts: number;
  combinatorial: boolean;
  prompts: string[];
  parsingError: string | undefined | null;
  isError: boolean;
  isLoading: boolean;
  seedBehaviour: SeedBehaviour;
}

export const initialDynamicPromptsState: DynamicPromptsState = {
  maxPrompts: 100,
  combinatorial: true,
  prompts: [],
  parsingError: undefined,
  isError: false,
  isLoading: false,
  seedBehaviour: 'PER_ITERATION',
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
    promptsChanged: (state, action: PayloadAction<string[]>) => {
      state.prompts = action.payload;
    },
    parsingErrorChanged: (state, action: PayloadAction<string | undefined>) => {
      state.parsingError = action.payload;
    },
    isErrorChanged: (state, action: PayloadAction<boolean>) => {
      state.isError = action.payload;
    },
    isLoadingChanged: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    seedBehaviourChanged: (state, action: PayloadAction<SeedBehaviour>) => {
      state.seedBehaviour = action.payload;
    },
  },
});

export const {
  maxPromptsChanged,
  maxPromptsReset,
  combinatorialToggled,
  promptsChanged,
  parsingErrorChanged,
  isErrorChanged,
  isLoadingChanged,
  seedBehaviourChanged,
} = dynamicPromptsSlice.actions;

export default dynamicPromptsSlice.reducer;
