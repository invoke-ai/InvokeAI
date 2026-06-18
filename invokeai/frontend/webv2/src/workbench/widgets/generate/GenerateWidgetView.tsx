import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

import {
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  isSupportedGenerateModel,
} from '@workbench/generation/baseGenerationPolicies';
import {
  isLoraModelConfig,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  syncGenerateWidgetValuesWithModels,
} from '@workbench/generation/settings';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useEffect, useMemo } from 'react';

import { GenerateSettingsForm } from './GenerateSettingsForm';

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
  const storedValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'generate'));
  const dispatch = useWorkbenchDispatch();
  const { error, models, status } = useModelsSnapshot();

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

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

    dispatch({ type: 'setGenerateSettings', values: { ...nextSettings, model } });
  }, [dispatch, models, selectedModel, settings, storedValues, supportedModels]);

  const commitSettings = (nextSettings: GenerateSettings) => {
    const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

    if (!model) {
      return;
    }

    dispatch({
      type: 'setGenerateSettings',
      values: { ...getSettingsWithAutoComponentSource(nextSettings, model, models), model },
    });
  };

  return (
    <GenerateSettingsForm
      isLoadingModels={status === 'idle' || status === 'loading'}
      loadError={error}
      loraModels={loraModels}
      selectedModel={selectedModel}
      settings={settings}
      supportedModels={supportedModels}
      vaeModels={vaeModels}
      onCommitSettings={commitSettings}
    />
  );
};
