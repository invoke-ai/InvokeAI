import type {
  GenerateModelConfig,
  GenerationModelCatalogItem,
  PromptHistoryItem,
} from '@features/generation/contracts';
import type { GenerateModelSelectionResult } from '@features/generation/settings';
import type { WorkbenchGenerationCommands } from '@workbench/workbenchStore';

import {
  getGenerateModelSelectionResult,
  getGenerateValuesRevertPatch,
  getPromptHistoryRecallPatch,
  isLossyModelSwitch,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';

export type ProjectGenerateModelSwitch =
  /** Nothing was written: the project already uses the model. */
  | { status: 'unchanged' }
  /** Nothing was written: the switch removes or resets something, so the model card asks first. */
  | { result: GenerateModelSelectionResult; status: 'needsConfirmation' }
  /** Written; `revertPatch` restores the earlier choices, or is null when there were none to restore. */
  | { result: GenerateModelSelectionResult; revertPatch: Record<string, unknown> | null; status: 'switched' };

/** The palette's model switch, with the model picker's rule: a switch that loses nothing applies at once. */
export const selectProjectGenerateModel = ({
  currentValues,
  generation,
  isCanvasFrame,
  model,
  models,
  projectId,
}: {
  currentValues: unknown;
  generation: Pick<WorkbenchGenerationCommands, 'setSettings'>;
  isCanvasFrame: boolean;
  model: GenerateModelConfig;
  models: readonly GenerationModelCatalogItem[];
  projectId: string;
}): ProjectGenerateModelSwitch => {
  const before = normalizeGenerateWidgetValues(currentValues);

  if (before?.model.key === model.key) {
    return { status: 'unchanged' };
  }

  const result = getGenerateModelSelectionResult({ currentValues, isCanvasFrame, model, models });

  if (isLossyModelSwitch(result)) {
    return { result, status: 'needsConfirmation' };
  }

  const next = { ...result.settings, model };

  generation.setSettings(next, projectId);

  return { result, revertPatch: before ? getGenerateValuesRevertPatch(before, next) : null, status: 'switched' };
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
