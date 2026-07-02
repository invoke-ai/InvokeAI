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
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useMemo } from 'react';

import { areProjectGenerateFormValuesEqual, getGenerateFormCommitPatch } from './generateFormViewModel';
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
  const storedSelection = useActiveProjectSelector(
    (project) => ({ projectId: project.id, values: getProjectWidgetValues(project, 'generate') }),
    areProjectGenerateFormValuesEqual
  );
  const projectId = storedSelection.projectId;
  const storedValues = storedSelection.values;
  const dispatch = useWorkbenchDispatch();
  const error = useModelsSelector((snapshot) => snapshot.error);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const status = useModelsSelector((snapshot) => snapshot.status);

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

    dispatch({
      projectId,
      type: 'patchGenerateSettings',
      values: getGenerateFormCommitPatch({ ...nextSettings, model }),
    });
  }, [dispatch, models, projectId, selectedModel, settings, storedValues, supportedModels]);

  const commitSettings = useCallback(
    (nextSettings: GenerateSettings) => {
      const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

      if (!model) {
        return;
      }

      dispatch({
        projectId,
        type: 'patchGenerateSettings',
        values: getGenerateFormCommitPatch({
          ...getSettingsWithAutoComponentSource(nextSettings, model, models),
          model,
        }),
      });
    },
    [dispatch, models, projectId, supportedModels]
  );

  const patchSettings = useCallback(
    (values: Partial<GenerateSettings>) => {
      dispatch({ projectId, type: 'patchGenerateSettings', values });
    },
    [dispatch, projectId]
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
