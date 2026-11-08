import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import {
  getDefaultGenerateSettings,
  isGenerateModelSelectable,
} from '@features/generation/core/baseGenerationPolicies';
import { isLoraModelConfig, normalizeGenerateSettings } from '@features/generation/core/settings';
import { useArchitectureCapabilitiesSelector } from '@features/generation/data/architectureCapabilitiesStore';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { GenerateSettingsForm } from './GenerateSettingsForm';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import { CapabilitiesGate } from './shared/CapabilitiesGate';

export const GenerateWidgetView = () => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const capabilitiesStatus = useArchitectureCapabilitiesSelector((snapshot) => snapshot.status);
  const projectId = ui.project.activeProjectId;
  const storedValues = useGenerateValues();
  const error = ui.models.error;
  const models = ui.models.catalog;
  const status = ui.models.status;

  const supportedModels = useMemo<GenerateModelConfig[]>(() => models.filter(isGenerateModelSelectable), [models]);
  const loraModels = useMemo(
    () => models.filter((model): model is ModelConfig & LoraModelConfig => isLoraModelConfig(model)),
    [models]
  );
  // Include capability revision/status so a cached null resolves again after loading.
  const resolved = useMemo(
    () => (capabilitiesStatus === 'loaded' ? resolveGenerateWidgetValues({ models, storedValues }) : null),
    [capabilitiesStatus, models, storedValues]
  );
  const settings =
    resolved?.values ?? normalizeGenerateSettings(storedValues) ?? getDefaultGenerateSettings(supportedModels[0]);
  const selectedModel = resolved?.values.model;
  const gateLabels = useMemo(
    () => ({
      loadFailed: t('widgets.generate.capabilitiesLoadFailed'),
      loading: t('widgets.generate.loadingCapabilities'),
      retry: t('widgets.generate.retry'),
    }),
    [t]
  );

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
    [models, projectId, supportedModels, ui.settings]
  );

  // Gate the form so edits cannot persist fallback architecture defaults.
  return (
    <CapabilitiesGate isReady={capabilitiesStatus === 'loaded'} labels={gateLabels}>
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
        onPatchSettings={ui.settings.patchGenerateSettings}
      />
    </CapabilitiesGate>
  );
};
