import type { GenerateModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type { Project } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { getPromptPolicy, isSupportedGenerateModel } from '@workbench/generation/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@workbench/generation/settings';
import { getProjectWidgetValues } from '@workbench/widgetState';

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

let navigationState: NavigationState | null = null;

export const resetPromptHistoryNavigation = (): void => {
  navigationState = null;
};

const getSelectedGenerateModel = (
  settings: ReturnType<typeof normalizeGenerateSettings>,
  models: readonly ModelConfig[] | undefined
): GenerateModelConfig | undefined => {
  if (!settings || !models) {
    return undefined;
  }

  return models.filter(isSupportedGenerateModel).find((model) => model.key === settings.modelKey);
};

const applyPromptHistoryIndex = ({
  dispatch,
  models,
  project,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  models: readonly ModelConfig[] | undefined;
  project: Project;
}): void => {
  if (!navigationState) {
    return;
  }

  const prompt = project.promptHistory[navigationState.historyIndex];
  const settings = normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'));

  if (!prompt || !settings) {
    return;
  }

  const selectedModel = getSelectedGenerateModel(settings, models);
  const promptPolicy = getPromptPolicy(selectedModel, settings);
  const values: Record<string, unknown> = { positivePrompt: prompt.positivePrompt };

  if (promptPolicy.negativeVisible) {
    values.negativePrompt = prompt.negativePrompt ?? '';
    values.negativePromptEnabled = prompt.negativePrompt ? true : settings.negativePromptEnabled;
  }

  dispatch({ projectId: project.id, type: 'patchWidgetValues', values, widgetId: 'generate' });
};

export const navigatePromptHistory = ({
  direction,
  dispatch,
  models,
  project,
}: {
  direction: -1 | 1;
  dispatch: Dispatch<WorkbenchAction>;
  models: readonly ModelConfig[] | undefined;
  project: Project;
}): void => {
  if (navigationState?.projectId !== project.id) {
    navigationState = null;
  }

  if (!isPositivePromptFocused() || project.promptHistory.length === 0) {
    return;
  }

  const settings = normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'));

  if (!settings) {
    return;
  }

  if (direction === -1) {
    if (!navigationState) {
      navigationState = {
        historyIndex: 0,
        projectId: project.id,
        stashedPrompts: {
          negativePrompt: settings.negativePrompt,
          negativePromptEnabled: settings.negativePromptEnabled,
          positivePrompt: settings.positivePrompt,
        },
      };
    } else if (navigationState.historyIndex < project.promptHistory.length - 1) {
      navigationState = { ...navigationState, historyIndex: navigationState.historyIndex + 1 };
    }

    applyPromptHistoryIndex({ dispatch, models, project });
    return;
  }

  if (!navigationState) {
    return;
  }

  const nextHistoryIndex = navigationState.historyIndex - 1;

  if (nextHistoryIndex < 0) {
    dispatch({
      projectId: project.id,
      type: 'patchWidgetValues',
      values: navigationState.stashedPrompts,
      widgetId: 'generate',
    });
    navigationState = null;
    return;
  }

  navigationState = { ...navigationState, historyIndex: nextHistoryIndex };
  applyPromptHistoryIndex({ dispatch, models, project });
};
