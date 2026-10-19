import type {
  GenerateModelConfig,
  GenerationModelCatalogItem,
  PromptHistoryItem,
} from '@features/generation/contracts';
import type { GenerateModelSelectionResult } from '@features/generation/settings';
import type { WorkbenchGenerationCommands } from '@workbench/workbenchStore';

import { getGenerateModelSelectionResult, getPromptHistoryRecallPatch } from '@features/generation/settings';

export const selectProjectGenerateModel = ({
  currentValues,
  generation,
  model,
  models,
  projectId,
}: {
  currentValues: unknown;
  generation: Pick<WorkbenchGenerationCommands, 'setSettings'>;
  model: GenerateModelConfig;
  models: readonly GenerationModelCatalogItem[];
  projectId: string;
}): GenerateModelSelectionResult => {
  const result = getGenerateModelSelectionResult({ currentValues, model, models });

  generation.setSettings({ ...result.settings, model }, projectId);

  return result;
};

export const recallProjectPromptHistoryItem = ({
  currentValues,
  generation,
  item,
  models,
  projectId,
}: {
  currentValues: unknown;
  generation: Pick<WorkbenchGenerationCommands, 'patchSettings'>;
  item: PromptHistoryItem;
  models: readonly GenerationModelCatalogItem[] | undefined;
  projectId: string;
}): boolean => {
  if (typeof currentValues !== 'object' || currentValues === null || Array.isArray(currentValues)) {
    return false;
  }

  const patch = getPromptHistoryRecallPatch({
    item,
    models,
    values: currentValues as Record<string, unknown>,
  });

  if (!patch) {
    return false;
  }

  generation.patchSettings(patch, projectId);
  return true;
};
