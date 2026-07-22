import type { GenerateModelConfig, GenerationModelCatalogItem } from '@features/generation/contracts';
import type { GenerateModelSelectionResult } from '@features/generation/settings';
import type { WorkbenchGenerationCommands } from '@workbench/workbenchStore';

import { getGenerateModelSelectionResult } from '@features/generation/settings';

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
