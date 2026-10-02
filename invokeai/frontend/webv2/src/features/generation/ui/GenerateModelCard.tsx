/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isGenerateModelSelectable,
} from '@features/generation/core/baseGenerationPolicies';
import { isGenerateModelConfig } from '@features/generation/core/settings';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';

const MAIN_MODEL_TYPES = ['main', 'external_image_generator'];

interface GenerateModelCardProps {
  isLoadingModels: boolean;
  loadError: string | null;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  settings: GenerateSettings;
  supportedModels: GenerateModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

/** This owner checks model availability; invocation readiness is validated elsewhere. */
export const GenerateModelCard = ({
  isLoadingModels,
  loadError,
  models,
  onCommitSettings,
  selectedModel,
  settings,
  supportedModels,
}: GenerateModelCardProps) => {
  const { i18n, t } = useTranslation();
  const ui = useGenerationUi();
  const { openManager } = ui.models;
  const openManagerForMainModels = useCallback(() => openManager({ modelType: 'main' }), [openManager]);
  // Retain only the selected model and recompute against live settings on confirmation.
  const [pendingSwitchModel, setPendingSwitchModel] = useState<GenerateModelConfig | null>(null);

  const pendingSwitchClearedLabels = useMemo(() => {
    if (!pendingSwitchModel) {
      return [];
    }

    return getGenerateModelSelectionResult({ currentValues: settings, model: pendingSwitchModel, models })
      .clearedLabels;
  }, [models, pendingSwitchModel, settings]);

  const commitModelSelection = (model: GenerateModelConfig) => {
    onCommitSettings(getGenerateModelSelectionResult({ currentValues: settings, model, models }).settings);
  };

  const selectModel = (model: GenerateModelConfig) => {
    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models });

    // Lossy switches confirm before committing; lossless ones stay instant.
    if (result.clearedLabels.length > 0) {
      setPendingSwitchModel(model);
      return;
    }

    onCommitSettings(result.settings);
  };

  const hasNoSupportedModels = !isLoadingModels && !loadError && supportedModels.length === 0;

  return (
    <Stack gap="1" py="1">
      <Field hint="model" label={t('widgets.generate.model')}>
        <ModelSelect
          filter={(model) => isGenerateModelConfig(model) && isGenerateModelSelectable(model)}
          isClearable={false}
          modelTypes={MAIN_MODEL_TYPES}
          placeholder={t('widgets.generate.selectModel')}
          value={selectedModel?.key ?? null}
          size="xs"
          onChange={(model) => {
            if (isGenerateModelConfig(model) && isGenerateModelSelectable(model)) {
              selectModel(model);
            }
          }}
        />
      </Field>

      {selectedModel ? null : isLoadingModels ? (
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.generate.loadingModels')}
        </Text>
      ) : loadError ? (
        <Text color="fg.error" fontSize="2xs">
          {loadError}
        </Text>
      ) : hasNoSupportedModels ? (
        <Stack gap="1.5">
          <Text color="fg.error" fontSize="2xs">
            {t('widgets.generate.noSupportedModels')}
          </Text>
          <Button alignSelf="flex-start" size="2xs" variant="outline" onClick={openManagerForMainModels}>
            {t('widgets.generate.openModelManager')}
          </Button>
        </Stack>
      ) : null}

      <ConfirmDialog
        body={
          <Text fontSize="sm">
            {t('widgets.generate.switchModelBody', {
              labels: new Intl.ListFormat(i18n.resolvedLanguage, { style: 'long', type: 'conjunction' }).format(
                pendingSwitchClearedLabels
              ),
              name: pendingSwitchModel?.name ?? '',
            })}
          </Text>
        }
        confirmLabel={t('widgets.generate.switchModelConfirm')}
        isOpen={pendingSwitchModel !== null}
        title={t('widgets.generate.switchModelTitle')}
        onClose={() => setPendingSwitchModel(null)}
        onConfirm={() => {
          if (pendingSwitchModel) {
            commitModelSelection(pendingSwitchModel);
          }
        }}
      />
    </Stack>
  );
};
