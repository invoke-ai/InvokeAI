import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import { getDefaultGenerateSettings, isSupportedGenerateModel } from '@features/generation/core/baseGenerationPolicies';
import { isLoraModelConfig, normalizeGenerateSettings } from '@features/generation/core/settings';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import { useCallback, useMemo } from 'react';

import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { GenerateSettingsForm } from './GenerateSettingsForm';
import { useGenerationUi } from './GenerationUiContext';

export const GenerateWidgetView = () => {
  const ui = useGenerationUi();
  const projectId = ui.project.activeProjectId;
  const storedValues = ui.project.generateValues;
  const error = ui.models.error;
  const models = ui.models.catalog;
  const status = ui.models.status;

  const supportedModels = useMemo<GenerateModelConfig[]>(() => models.filter(isSupportedGenerateModel), [models]);
  const loraModels = useMemo(
    () => models.filter((model): model is ModelConfig & LoraModelConfig => isLoraModelConfig(model)),
    [models]
  );
  const resolved = useMemo(() => resolveGenerateWidgetValues({ models, storedValues }), [models, storedValues]);
  const settings =
    resolved?.values ?? normalizeGenerateSettings(storedValues) ?? getDefaultGenerateSettings(supportedModels[0]);
  const selectedModel = resolved?.values.model;

  const commitSettings = useCallback(
    (nextSettings: GenerateSettings) => {
      const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

      if (!model) {
        return;
      }

      const next = resolveGenerateWidgetValues({
        models,
        storedValues: { ...nextSettings, model },
      });

      if (next) {
        ui.settings.patchGenerateSettings(getGenerateFormCommitPatch(next.values), projectId);
      }
    },
    [models, projectId, supportedModels, ui]
  );

  const patchSettings = useCallback(
    (values: Partial<GenerateSettings>) => {
      ui.settings.patchGenerateSettings(values, projectId);
    },
    [projectId, ui]
  );

  return (
    <GenerateSettingsForm
      isLoadingModels={status === 'idle' || status === 'loading'}
      loadError={error}
      loraModels={loraModels}
      models={models}
      projectId={projectId}
      selectedModel={selectedModel}
      settings={settings}
      supportedModels={supportedModels}
      onCommitSettings={commitSettings}
      onPatchSettings={patchSettings}
    />
  );
};
