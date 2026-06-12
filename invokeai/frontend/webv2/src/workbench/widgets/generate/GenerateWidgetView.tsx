import { useEffect, useMemo } from 'react';

import { getDefaultGenerateSettings, isSupportedGenerateModel } from '../../generation/graph';
import { normalizeGenerateSettings, normalizeGenerateWidgetValues } from '../../generation/settings';
import type { GenerateSettings, VaeModelConfig } from '../../generation/types';
import { ensureModelsLoaded, useModelsSnapshot } from '../../models/modelsStore';
import type { ModelConfig } from '../../models/types';
import { useWorkbench } from '../../WorkbenchContext';
import { GenerateSettingsForm } from './GenerateSettingsForm';

export const GenerateWidgetView = () => {
  const { activeProject, dispatch } = useWorkbench();
  const { error, models, status } = useModelsSnapshot();

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const storedValues = activeProject.widgetStates.generate.values;
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
