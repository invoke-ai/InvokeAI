/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@features/generation/core/types';

import { Badge, HStack, Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  getGenerationDimensions,
  getMaxReferenceImages,
  getPromptPolicy,
  isGenerateModelSelectable,
  isReferenceImageSupported,
} from '@features/generation/core/baseGenerationPolicies';
import { isGenerateModelConfig, isVaeModelConfig } from '@features/generation/core/settings';
import { Button, ConfirmDialog, Field } from '@platform/ui';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { countModelDefaultOverrides, getModelDefaultSettings } from './shared/modelDefaultSettings';

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

/**
 * The panel's tier-1 surface: the model choice, what that model offers, and
 * every model-availability state — loading, backend errors, and the empty
 * catalog — local to the choice that causes them. Readiness/validation is
 * deliberately absent here (the invocation controls own it), and the preset
 * library and reset-all live in the widget header.
 */
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
  // A switch that clears incompatible settings waits behind a confirm. Only the
  // model is held; the transition is recomputed against live settings on
  // confirm, and the labels shown come from a preview run of the same move.
  const [pendingSwitchModel, setPendingSwitchModel] = useState<GenerateModelConfig | null>(null);

  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => isVaeModelConfig(model)),
    [models]
  );

  const overrideCount = useMemo(() => {
    if (!selectedModel) {
      return 0;
    }

    return countModelDefaultOverrides(settings, getModelDefaultSettings(settings, selectedModel, vaeModels));
  }, [selectedModel, settings, vaeModels]);

  /** What the model offers, in one quiet line: native size, then capabilities. */
  const features = useMemo(() => {
    if (!selectedModel) {
      return [];
    }

    const entries: string[] = [
      t('widgets.generate.nativePx', { size: getGenerationDimensions(selectedModel).optimal }),
    ];

    if (getPromptPolicy(selectedModel, settings).negativeVisible) {
      entries.push(t('widgets.generate.negativePrompt'));
    }

    if (isReferenceImageSupported(selectedModel)) {
      entries.push(t('widgets.generate.referencesMax', { count: getMaxReferenceImages(selectedModel) }));
    }

    if (selectedModel.type !== 'external_image_generator') {
      entries.push(t('widgets.generate.concepts'));
    }

    return entries;
  }, [selectedModel, settings, t]);

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

      {selectedModel ? (
        <HStack gap="2" justify="space-between">
          <Text color="fg.subtle" fontSize="2xs" minW="0">
            {features.join(' · ')}
          </Text>
          {overrideCount > 0 ? (
            <Badge flexShrink="0" size="xs" variant="surface">
              {t('widgets.generate.overridesCount', { count: overrideCount })}
            </Badge>
          ) : null}
        </HStack>
      ) : isLoadingModels ? (
        <Text color="fg.subtle" fontSize="2xs">
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
          <Button alignSelf="flex-start" size="2xs" variant="outline" onClick={openManager}>
            {t('widgets.generate.openModelManager')}
          </Button>
        </Stack>
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.generate.chooseModelToStart')}
        </Text>
      )}

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
