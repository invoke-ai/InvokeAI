/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isGenerateModelSelectable,
} from '@features/generation/core/baseGenerationPolicies';
import { isGenerateModelConfig } from '@features/generation/core/settings';
import { areArraysEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateDraft } from './generateDebounce';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { notifyGenerateDimensionsAdjusted } from './modelSelectionNotice';

const MAIN_MODEL_TYPES = ['main', 'external_image_generator'];

/** What a pending switch would change in the live draft: settings it removes, and the size it moves to, if any. */
interface PendingSwitchPreview {
  clearedLabels: readonly string[];
  adjustedSize: { height: number; width: number } | null;
}

const NO_PENDING_SWITCH: PendingSwitchPreview = { adjustedSize: null, clearedLabels: [] };

const arePreviewsEqual = (left: PendingSwitchPreview, right: PendingSwitchPreview) =>
  areArraysEqual(left.clearedLabels, right.clearedLabels) &&
  left.adjustedSize?.width === right.adjustedSize?.width &&
  left.adjustedSize?.height === right.adjustedSize?.height;

interface GenerateModelCardProps {
  draft: GenerateDraft;
  isLoadingModels: boolean;
  loadError: string | null;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  supportedModels: GenerateModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

/** This owner checks model availability; invocation readiness is validated elsewhere. */
export const GenerateModelCard = ({
  draft,
  isLoadingModels,
  loadError,
  models,
  onCommitSettings,
  selectedModel,
  supportedModels,
}: GenerateModelCardProps) => {
  const { i18n, t } = useTranslation();
  const ui = useGenerationUi();
  const { openManager } = ui.models;
  const openManagerForMainModels = useCallback(() => openManager({ modelType: 'main' }), [openManager]);
  // Retain only the selected model and recompute against the current draft on confirmation.
  const [pendingSwitchModel, setPendingSwitchModel] = useState<GenerateModelConfig | null>(null);

  // Constant until a switch awaits confirmation; then the open dialog lists what the latest draft would lose.
  const pendingSwitch = useExternalStoreSelector(
    draft.subscribe,
    draft.getSnapshot,
    (settings: GenerateSettings): PendingSwitchPreview => {
      if (!pendingSwitchModel) {
        return NO_PENDING_SWITCH;
      }

      const result = getGenerateModelSelectionResult({ currentValues: settings, model: pendingSwitchModel, models });

      return {
        adjustedSize: result.dimensionsAdjusted
          ? { height: result.settings.height, width: result.settings.width }
          : null,
        clearedLabels: result.clearedLabels,
      };
    },
    arePreviewsEqual
  );

  const commitModelSelection = (model: GenerateModelConfig) => {
    onCommitSettings(getGenerateModelSelectionResult({ currentValues: draft.getSnapshot(), model, models }).settings);
  };

  const selectModel = (model: GenerateModelConfig) => {
    const result = getGenerateModelSelectionResult({ currentValues: draft.getSnapshot(), model, models });

    // Lossy switches confirm before committing; lossless ones stay instant and report a moved size.
    if (result.clearedLabels.length > 0) {
      setPendingSwitchModel(model);
      return;
    }

    onCommitSettings(result.settings);
    notifyGenerateDimensionsAdjusted({ modelName: model.name, notifications: ui.notifications, result, t });
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
          onChange={(model) => {
            if (isGenerateModelConfig(model) && isGenerateModelSelectable(model)) {
              selectModel(model);
            }
          }}
        />
      </Field>

      {selectedModel ? null : isLoadingModels ? (
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.generate.loadingModels')}
        </Text>
      ) : loadError ? (
        <Text color="fg.error" fontSize="xs">
          {loadError}
        </Text>
      ) : hasNoSupportedModels ? (
        <Stack gap="1.5">
          <Text color="fg.error" fontSize="xs">
            {t('widgets.generate.noSupportedModels')}
          </Text>
          <Button alignSelf="flex-start" size="sm" variant="outline" onClick={openManagerForMainModels}>
            {t('widgets.generate.openModelManager')}
          </Button>
        </Stack>
      ) : null}

      <ConfirmDialog
        body={
          <>
            <Text fontSize="lg">
              {t('widgets.generate.switchModelBody', {
                labels: new Intl.ListFormat(i18n.resolvedLanguage, { style: 'long', type: 'conjunction' }).format(
                  pendingSwitch.clearedLabels
                ),
                name: pendingSwitchModel?.name ?? '',
              })}
            </Text>
            {pendingSwitch.adjustedSize ? (
              <Text color="fg.muted" fontSize="md">
                {t('widgets.generate.switchModelAdjustsDimensions', pendingSwitch.adjustedSize)}
              </Text>
            ) : null}
          </>
        }
        confirmLabel={t('widgets.generate.switchModelConfirm')}
        isOpen={pendingSwitchModel !== null}
        title={t('widgets.generate.switchModelTitle')}
        onClose={() => setPendingSwitchModel(null)}
        onConfirm={() => {
          if (pendingSwitchModel) {
            // Close before committing, so the dialog animates out listing what was confirmed, not the switched draft.
            setPendingSwitchModel(null);
            commitModelSelection(pendingSwitchModel);
          }
        }}
      />
    </Stack>
  );
};
