import { useEffect, useMemo, useState } from 'react';

import { listMainModels } from '../../generation/api';
import { getDefaultGenerateSettings, isSupportedGenerateModel } from '../../generation/graph';
import type { GenerateSettings, MainModelConfig } from '../../generation/types';
import { useWorkbench } from '../../WorkbenchContext';
import { GenerateSettingsForm } from './GenerateSettingsForm';
import { isGenerateSettings, isGenerateWidgetValues } from './generateWidgetGuards';

export const GenerateWidgetView = () => {
  const { activeProject, dispatch } = useWorkbench();
  const [models, setModels] = useState<MainModelConfig[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const storedValues = activeProject.widgetStates.generate.values;
  const selectedModel = supportedModels.find(
    (model) => model.key === activeProject.widgetStates.generate.values.modelKey
  );
  const settings = isGenerateSettings(storedValues)
    ? storedValues
    : getDefaultGenerateSettings(selectedModel ?? supportedModels[0]);

  useEffect(() => {
    let isStale = false;

    listMainModels()
      .then((nextModels) => {
        if (isStale) {
          return;
        }

        setModels(nextModels);
        setLoadError(null);
      })
      .catch((error: unknown) => {
        if (isStale) {
          return;
        }

        const message = error instanceof Error ? error.message : 'Failed to load models.';
        setLoadError(message);
        dispatch({ message, type: 'recordError' });
      })
      .finally(() => {
        if (!isStale) {
          setIsLoadingModels(false);
        }
      });

    return () => {
      isStale = true;
    };
  }, [dispatch]);

  useEffect(() => {
    const model = selectedModel ?? supportedModels[0];

    if (!model) {
      return;
    }

    if (isGenerateWidgetValues(storedValues) && storedValues.model.key === model.key) {
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
      isLoadingModels={isLoadingModels}
      loadError={loadError}
      settings={settings}
      supportedModels={supportedModels}
      onCommitSettings={commitSettings}
    />
  );
};
