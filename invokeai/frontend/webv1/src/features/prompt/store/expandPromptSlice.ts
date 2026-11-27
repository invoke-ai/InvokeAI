import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { systemPromptsApi } from 'services/api/endpoints/systemPrompts';
import { assert } from 'tsafe';
import z from 'zod';

const zExpandPromptState = z.object({
  selectedSystemPromptId: z.string().nullable(),
  selectedModelKey: z.string().nullable(),
});

type ExpandPromptState = z.infer<typeof zExpandPromptState>;

const getInitialState = (): ExpandPromptState => ({
  selectedSystemPromptId: null,
  selectedModelKey: null,
});

const slice = createSlice({
  name: 'expandPrompt',
  initialState: getInitialState(),
  reducers: {
    selectedSystemPromptIdChanged: (state, action: PayloadAction<string | null>) => {
      state.selectedSystemPromptId = action.payload;
    },
    selectedModelKeyChanged: (state, action: PayloadAction<string | null>) => {
      state.selectedModelKey = action.payload;
    },
  },
  extraReducers(builder) {
    // If the selected prompt has been deleted on the server, clear the local selection
    // so the picker doesn't show a stale ID.
    builder.addMatcher(systemPromptsApi.endpoints.deleteSystemPrompt.matchFulfilled, (state, action) => {
      if (state.selectedSystemPromptId === action.meta.arg.originalArgs) {
        state.selectedSystemPromptId = null;
      }
    });
    builder.addMatcher(systemPromptsApi.endpoints.listSystemPrompts.matchFulfilled, (state, action) => {
      if (state.selectedSystemPromptId === null) {
        return;
      }
      const ids = action.payload.map((p) => p.id);
      if (!ids.includes(state.selectedSystemPromptId)) {
        state.selectedSystemPromptId = null;
      }
    });
  },
});

export const { selectedSystemPromptIdChanged, selectedModelKeyChanged } = slice.actions;

export const expandPromptSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zExpandPromptState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zExpandPromptState.parse(state);
    },
  },
};

const selectExpandPromptSlice = (state: RootState) => state.expandPrompt;
const createExpandPromptSelector = <T>(selector: Selector<ExpandPromptState, T>) =>
  createSelector(selectExpandPromptSlice, selector);

export const selectSelectedSystemPromptId = createExpandPromptSelector((s) => s.selectedSystemPromptId);
export const selectSelectedModelKey = createExpandPromptSelector((s) => s.selectedModelKey);
