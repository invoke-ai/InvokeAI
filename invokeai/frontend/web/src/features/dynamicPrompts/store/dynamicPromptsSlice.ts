import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { buildZodTypeGuard } from 'common/util/zodUtils';
import { isPlainObject } from 'es-toolkit';
import { assert } from 'tsafe';
import { z } from 'zod';

const zSeedBehaviour = z.enum(['PER_ITERATION', 'PER_PROMPT']);
export const isSeedBehaviour = buildZodTypeGuard(zSeedBehaviour);
export type SeedBehaviour = z.infer<typeof zSeedBehaviour>;

const zDynamicPromptsState = z.object({
  _version: z.literal(1),
  maxPrompts: z.number().int().min(1).max(1000),
  combinatorial: z.boolean(),
  prompts: z.array(z.string()),
  parsingError: z.string().nullish(),
  isError: z.boolean(),
  isLoading: z.boolean(),
  seedBehaviour: zSeedBehaviour,
});
export type DynamicPromptsState = z.infer<typeof zDynamicPromptsState>;

const getInitialState = (): DynamicPromptsState => ({
  _version: 1,
  maxPrompts: 100,
  combinatorial: true,
  prompts: [],
  parsingError: undefined,
  isError: false,
  isLoading: false,
  seedBehaviour: 'PER_ITERATION',
});

const slice = createSlice({
  name: 'dynamicPrompts',
  initialState: getInitialState(),
  reducers: {
    maxPromptsChanged: (state, action: PayloadAction<number>) => {
      state.maxPrompts = action.payload;
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
  promptsChanged,
  parsingErrorChanged,
  isErrorChanged,
  isLoadingChanged,
  seedBehaviourChanged,
} = slice.actions;

export const dynamicPromptsSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zDynamicPromptsState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zDynamicPromptsState.parse(state);
    },
    persistDenylist: ['prompts', 'parsingError', 'isError', 'isLoading'],
  },
};

export const selectDynamicPromptsSlice = (state: RootState) => state.dynamicPrompts;
const createDynamicPromptsSelector = <T>(selector: Selector<DynamicPromptsState, T>) =>
  createSelector(selectDynamicPromptsSlice, selector);

export const selectDynamicPromptsMaxPrompts = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.maxPrompts
);
export const selectDynamicPromptsCombinatorial = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.combinatorial
);
export const selectDynamicPromptsPrompts = createDynamicPromptsSelector((dynamicPrompts) => dynamicPrompts.prompts);
export const selectDynamicPromptsParsingError = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.parsingError
);
export const selectDynamicPromptsIsError = createDynamicPromptsSelector((dynamicPrompts) => dynamicPrompts.isError);
export const selectDynamicPromptsIsLoading = createDynamicPromptsSelector((dynamicPrompts) => dynamicPrompts.isLoading);
export const selectDynamicPromptsSeedBehaviour = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.seedBehaviour
);
