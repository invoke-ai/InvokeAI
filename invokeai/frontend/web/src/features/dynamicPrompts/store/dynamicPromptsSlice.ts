import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { z } from 'zod';

const zSeedBehaviour = z.enum(['PER_ITERATION', 'PER_PROMPT']);
type SeedBehaviour = z.infer<typeof zSeedBehaviour>;
export const isSeedBehaviour = (v: unknown): v is SeedBehaviour => zSeedBehaviour.safeParse(v).success;

interface DynamicPromptsState {
  _version: 1;
  maxPrompts: number;
  combinatorial: boolean;
  prompts: string[];
  parsingError: string | undefined | null;
  isError: boolean;
  isLoading: boolean;
  seedBehaviour: SeedBehaviour;
}

const initialDynamicPromptsState: DynamicPromptsState = {
  _version: 1,
  maxPrompts: 100,
  combinatorial: true,
  prompts: [],
  parsingError: undefined,
  isError: false,
  isLoading: false,
  seedBehaviour: 'PER_ITERATION',
};

export const dynamicPromptsSlice = createSlice({
  name: 'dynamicPrompts',
  initialState: initialDynamicPromptsState,
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
      state.isLoading = false;
    },
    parsingErrorChanged: (state, action: PayloadAction<string | null | undefined>) => {
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

export const selectDynamicPromptsSlice = (state: RootState) => state.dynamicPrompts;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateDynamicPromptsState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const dynamicPromptsPersistConfig: PersistConfig<DynamicPromptsState> = {
  name: dynamicPromptsSlice.name,
  initialState: initialDynamicPromptsState,
  migrate: migrateDynamicPromptsState,
  persistDenylist: ['prompts'],
};
