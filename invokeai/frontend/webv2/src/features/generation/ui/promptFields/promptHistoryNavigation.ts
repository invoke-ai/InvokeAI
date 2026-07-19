import type { GenerationModelCatalogItem as ModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { GenerateModelConfig } from '@features/generation/core/types';

import { getPromptPolicy, isSupportedGenerateModel } from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';

import { isPositivePromptFocused } from './promptFocus';

type NavigationState = {
  projectId: string;
  historyIndex: number;
  stashedPrompts: {
    negativePrompt: string;
    negativePromptEnabled: boolean;
    positivePrompt: string;
  };
};

interface NavigateOptions {
  direction: -1 | 1;
  patchValues: (values: Record<string, unknown>, projectId: string) => void;
  models: readonly ModelConfig[] | undefined;
  projectId: string;
  promptHistory: readonly PromptHistoryItem[];
  values: Record<string, unknown>;
}

export interface PromptHistoryNavigation {
  navigate(options: NavigateOptions): void;
  reset(): void;
}

const getSelectedGenerateModel = (
  settings: ReturnType<typeof normalizeGenerateSettings>,
  models: readonly ModelConfig[] | undefined
): GenerateModelConfig | undefined => {
  if (!settings || !models) {
    return undefined;
  }

  return models.filter(isSupportedGenerateModel).find((model) => model.key === settings.modelKey);
};

export const createPromptHistoryNavigation = (): PromptHistoryNavigation => {
  let state: NavigationState | null = null;

  const applyHistoryIndex = ({
    patchValues,
    models,
    projectId,
    promptHistory,
    values,
  }: Omit<NavigateOptions, 'direction'>): void => {
    if (!state) {
      return;
    }

    const prompt = promptHistory[state.historyIndex];
    const settings = normalizeGenerateSettings(values);

    if (!prompt || !settings) {
      return;
    }

    const selectedModel = getSelectedGenerateModel(settings, models);
    const promptPolicy = getPromptPolicy(selectedModel, settings);
    const patch: Record<string, unknown> = { positivePrompt: prompt.positivePrompt };

    if (promptPolicy.negativeVisible) {
      patch.negativePrompt = prompt.negativePrompt ?? '';
      patch.negativePromptEnabled = prompt.negativePrompt ? true : settings.negativePromptEnabled;
    }

    patchValues(patch, projectId);
  };

  return {
    navigate({ direction, patchValues, models, projectId, promptHistory, values }) {
      if (state?.projectId !== projectId) {
        state = null;
      }

      if (!isPositivePromptFocused() || promptHistory.length === 0) {
        return;
      }

      const settings = normalizeGenerateSettings(values);

      if (!settings) {
        return;
      }

      if (direction === -1) {
        if (!state) {
          state = {
            historyIndex: 0,
            projectId,
            stashedPrompts: {
              negativePrompt: settings.negativePrompt,
              negativePromptEnabled: settings.negativePromptEnabled,
              positivePrompt: settings.positivePrompt,
            },
          };
        } else if (state.historyIndex < promptHistory.length - 1) {
          state = { ...state, historyIndex: state.historyIndex + 1 };
        }

        applyHistoryIndex({ models, patchValues, projectId, promptHistory, values });
        return;
      }

      if (!state) {
        return;
      }

      const nextHistoryIndex = state.historyIndex - 1;

      if (nextHistoryIndex < 0) {
        patchValues(state.stashedPrompts, projectId);
        state = null;
        return;
      }

      state = { ...state, historyIndex: nextHistoryIndex };
      applyHistoryIndex({ models, patchValues, projectId, promptHistory, values });
    },
    reset() {
      state = null;
    },
  };
};

/** One navigation cursor per window: the hotkey commands and both prompt fields must share it. */
export const promptHistoryNavigation: PromptHistoryNavigation = createPromptHistoryNavigation();
