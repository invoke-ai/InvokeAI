import type { GenerateSettings, VaeModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

import { getDefaultGenerateSettings, isSupportedGenerateModel } from '@workbench/generation/graph';
import { normalizeGenerateSettings, normalizeGenerateWidgetValues } from '@workbench/generation/settings';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useEffect, useMemo } from 'react';

import { GenerateSettingsForm } from './GenerateSettingsForm';

export const GenerateWidgetView = () => {
  const storedValues = useActiveProjectSelector((project) => project.widgetStates.generate.values);
  const dispatch = useWorkbenchDispatch();
  const { error, models, status } = useModelsSnapshot();

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
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

    if (storedWidgetValues && storedWidgetValues.model.key === model.key) {
      return;
    }

    const nextSettings = selectedModel ? settings : getDefaultGenerateSettings(model);

    dispatch({ type: 'setGenerateSettings', values: { ...nextSettings, model } });
  }, [dispatch, selectedModel, settings, storedValues, supportedModels]);

  const commitSettings = (nextSettings: GenerateSettings) => {
    const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

    if (!model) {
      return;
    }

    dispatch({ type: 'setGenerateSettings', values: { ...nextSettings, model } });
  };

  return (
    <GenerateSettingsForm
      isLoadingModels={status === 'idle' || status === 'loading'}
      loadError={error}
      selectedModel={selectedModel}
      settings={settings}
      supportedModels={supportedModels}
      vaeModels={vaeModels}
      onCommitSettings={commitSettings}
    />
  );
};
