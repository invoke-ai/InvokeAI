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

const zDynamicPromptMode = z.enum(['random', 'combinatorial']);
export const isDynamicPromptMode = buildZodTypeGuard(zDynamicPromptMode);
export type DynamicPromptMode = z.infer<typeof zDynamicPromptMode>;

const zDynamicPromptRandomRefreshMode = z.enum(['manual', 'per_enqueue']);
export const isDynamicPromptRandomRefreshMode = buildZodTypeGuard(zDynamicPromptRandomRefreshMode);
export type DynamicPromptRandomRefreshMode = z.infer<typeof zDynamicPromptRandomRefreshMode>;

const zDynamicPromptsState = z.object({
  _version: z.literal(3),
  mode: zDynamicPromptMode,
  randomSamples: z.number().int().min(1).max(1000),
  maxCombinations: z.number().int().min(1).max(10000),
  randomSeed: z.number().int().min(0),
  randomRefreshMode: zDynamicPromptRandomRefreshMode,
  prompts: z.array(z.string()),
  parsingError: z.string().nullish(),
  isError: z.boolean(),
  isLoading: z.boolean(),
  seedBehaviour: zSeedBehaviour,
});
export type DynamicPromptsState = z.infer<typeof zDynamicPromptsState>;

const getInitialState = (): DynamicPromptsState => ({
  _version: 3,
  mode: 'random',
  randomSamples: 1,
  maxCombinations: 100,
  randomSeed: 0,
  randomRefreshMode: 'per_enqueue',
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
    modeChanged: (state, action: PayloadAction<DynamicPromptMode>) => {
      state.mode = action.payload;
    },
    randomSamplesChanged: (state, action: PayloadAction<number>) => {
      state.randomSamples = action.payload;
    },
    maxCombinationsChanged: (state, action: PayloadAction<number>) => {
      state.maxCombinations = action.payload;
    },
    randomSeedChanged: (state, action: PayloadAction<number>) => {
      state.randomSeed = action.payload;
    },
    randomRefreshModeChanged: (state, action: PayloadAction<DynamicPromptRandomRefreshMode>) => {
      state.randomRefreshMode = action.payload;
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
  modeChanged,
  randomSamplesChanged,
  maxCombinationsChanged,
  randomSeedChanged,
  randomRefreshModeChanged,
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
      const initialState = getInitialState();
      if (state._version === 2) {
        const mode = isDynamicPromptMode(state.mode) ? state.mode : initialState.mode;
        return zDynamicPromptsState.parse({
          ...initialState,
          ...state,
          _version: 3,
          mode,
          randomRefreshMode:
            mode === 'random' && state.randomRefreshMode !== 'manual' ? 'per_enqueue' : 'manual',
          seedBehaviour: isSeedBehaviour(state.seedBehaviour) ? state.seedBehaviour : initialState.seedBehaviour,
        });
      }
      if (state._version !== 3) {
        const legacyCombinatorial = state.combinatorial === true;
        const legacyMaxPrompts = typeof state.maxPrompts === 'number' ? state.maxPrompts : initialState.maxCombinations;
        return zDynamicPromptsState.parse({
          ...initialState,
          mode: legacyCombinatorial ? 'combinatorial' : 'random',
          maxCombinations: legacyMaxPrompts,
          randomRefreshMode: legacyCombinatorial ? 'manual' : 'per_enqueue',
          seedBehaviour: isSeedBehaviour(state.seedBehaviour) ? state.seedBehaviour : initialState.seedBehaviour,
        });
      }
      return zDynamicPromptsState.parse({ ...initialState, ...state });
    },
    persistDenylist: ['prompts', 'parsingError', 'isError', 'isLoading'],
  },
};

export const selectDynamicPromptsSlice = (state: RootState) => state.dynamicPrompts;
const createDynamicPromptsSelector = <T>(selector: Selector<DynamicPromptsState, T>) =>
  createSelector(selectDynamicPromptsSlice, selector);

export const selectDynamicPromptsMode = createDynamicPromptsSelector((dynamicPrompts) => dynamicPrompts.mode);
export const selectDynamicPromptsRandomSamples = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.randomSamples
);
export const selectDynamicPromptsMaxCombinations = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.maxCombinations
);
export const selectDynamicPromptsRandomSeed = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.randomSeed
);
export const selectDynamicPromptsRandomRefreshMode = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.randomRefreshMode
);
export const selectDynamicPromptsMaxPrompts = createDynamicPromptsSelector(
  (dynamicPrompts) =>
    dynamicPrompts.mode === 'combinatorial' ? dynamicPrompts.maxCombinations : dynamicPrompts.randomSamples
);
export const selectDynamicPromptsCombinatorial = createDynamicPromptsSelector(
  (dynamicPrompts) => dynamicPrompts.mode === 'combinatorial'
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
