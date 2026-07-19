import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@features/generation/core/types';

import {
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  isSupportedGenerateModel,
} from '@features/generation/core/baseGenerationPolicies';
import {
  isLoraModelConfig,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  syncGenerateWidgetValuesWithModels,
} from '@features/generation/core/settings';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useCallback, useEffect, useMemo } from 'react';

import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { GenerateSettingsForm } from './GenerateSettingsForm';
import { useGenerationUi } from './GenerationUiContext';

const getSettingsWithAutoComponentSource = (
  nextSettings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): GenerateSettings => {
  const componentSourceModel = getAutoFlux2ComponentSourceModel(model, nextSettings, models);

  if (componentSourceModel === undefined || componentSourceModel?.key === nextSettings.componentSourceModel?.key) {
    return nextSettings;
  }

  return { ...nextSettings, componentSourceModel };
};

export const GenerateWidgetView = () => {
  const ui = useGenerationUi();
  const projectId = ui.project.activeProjectId;
  const storedValues = ui.project.generateValues;
  const error = ui.models.error;
  const models = ui.models.catalog;
  const status = ui.models.status;

  useMountEffect(() => {
    ui.models.ensureLoaded();
  });

  const supportedModels = useMemo<GenerateModelConfig[]>(() => models.filter(isSupportedGenerateModel), [models]);
  const loraModels = useMemo(
    () => models.filter((model): model is ModelConfig & LoraModelConfig => isLoraModelConfig(model)),
    [models]
  );
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const normalizedSettings = normalizeGenerateSettings(storedValues);
  const selectedModel = supportedModels.find((model) => model.key === normalizedSettings?.modelKey);
  const settings = normalizedSettings ?? getDefaultGenerateSettings(selectedModel ?? supportedModels[0]);

  useEffect(() => {
    const model = selectedModel ?? supportedModels[0];

    if (!model) {
      return;
    }

    const storedWidgetValues = normalizeGenerateWidgetValues(storedValues);
    const syncedWidgetValues = storedWidgetValues
      ? syncGenerateWidgetValuesWithModels(storedWidgetValues, models)
      : null;
    const baseSettings = syncedWidgetValues ?? (selectedModel ? settings : getDefaultGenerateSettings(model));
    const nextSettings = getSettingsWithAutoComponentSource(baseSettings, model, models);

    if (
      storedWidgetValues &&
      storedWidgetValues.model.key === model.key &&
      syncedWidgetValues === storedWidgetValues &&
      nextSettings === baseSettings
    ) {
      return;
    }

    ui.settings.patchGenerateSettings(getGenerateFormCommitPatch({ ...nextSettings, model }), projectId);
  }, [models, projectId, selectedModel, settings, storedValues, supportedModels, ui]);

  const commitSettings = useCallback(
    (nextSettings: GenerateSettings) => {
      const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

      if (!model) {
        return;
      }

      ui.settings.patchGenerateSettings(
        getGenerateFormCommitPatch({
          ...getSettingsWithAutoComponentSource(nextSettings, model, models),
          model,
        }),
        projectId
      );
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
      vaeModels={vaeModels}
      onCommitSettings={commitSettings}
      onPatchSettings={patchSettings}
    />
  );
};
