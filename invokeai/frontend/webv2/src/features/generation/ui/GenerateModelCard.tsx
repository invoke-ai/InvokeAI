/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isGenerateModelSelectable,
  isLossyModelSwitch,
} from '@features/generation/core/baseGenerationPolicies';
import { describeGenerationIssue } from '@features/generation/core/generationIssues';
import { getGenerateValuesRevertPatch, isGenerateModelConfig } from '@features/generation/core/settings';
import { areArraysEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { useCallback, useId } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateDraft } from './generateDebounce';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import {
  clearGenerateModelSwitchConfirmation,
  describeModelSwitchChanges,
  notifyGenerateModelSwitched,
  requestGenerateModelSwitchConfirmation,
  usePendingGenerateModelSwitch,
} from './modelSelectionNotice';
import { useGenerateSectionIssues } from './shared/GenerateIssues';

const MAIN_MODEL_TYPES = ['main', 'external_image_generator'];

const NO_LINES: string[] = [];

interface GenerateModelCardProps {
  draft: GenerateDraft;
  isLoadingModels: boolean;
  loadError: string | null;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  supportedModels: GenerateModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

/**
 * This owner picks the model and reports it missing; invocation readiness is validated elsewhere. A switch that only
 * sets choices aside applies at once with Undo; one that removes or resets something asks first, then offers Undo too.
 */
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
  const projectId = ui.project.activeProjectId;
  const selectId = useId();
  // On a Canvas source the width and height are the frame, which a switch keeps for a hosted provider.
  const isCanvasFrame = ui.project.invocationSourceId === 'canvas';
  const openManagerForMainModels = useCallback(() => openManager({ modelType: 'main' }), [openManager]);
  // Shared with the palette, which asks this card to confirm a lossy switch it cannot show itself.
  const pendingSwitchModel = usePendingGenerateModelSwitch(projectId);
  const modelIssues = useGenerateSectionIssues('model');
  const isMissing =
    selectedModel !== undefined && models.length > 0 && !models.some((model) => model.key === selectedModel.key);

  // Constant until a switch awaits confirmation; then the open dialog lists what the latest draft would change.
  const pendingLines = useExternalStoreSelector(
    draft.subscribe,
    draft.getSnapshot,
    (settings: GenerateSettings): string[] =>
      pendingSwitchModel
        ? describeModelSwitchChanges(
            getGenerateModelSelectionResult({
              currentValues: settings,
              isCanvasFrame,
              model: pendingSwitchModel,
              models,
            }),
            t,
            i18n.resolvedLanguage,
            'widgets.generate.switchModelLines'
          )
        : NO_LINES,
    areArraysEqual
  );

  const commitModelSelection = (model: GenerateModelConfig) => {
    const before = draft.getSnapshot();
    const result = getGenerateModelSelectionResult({ currentValues: before, isCanvasFrame, model, models });

    onCommitSettings(result.settings);

    // A first pick has nothing to go back to.
    if (selectedModel) {
      const revertPatch = getGenerateValuesRevertPatch(
        { ...before, model: selectedModel },
        { ...result.settings, model }
      );

      notifyGenerateModelSwitched({
        getCurrentModelKey: () => {
          const modelKey = ui.settings.readGenerateValues(projectId)?.modelKey;

          return typeof modelKey === 'string' ? modelKey : undefined;
        },
        locale: i18n.resolvedLanguage,
        model,
        notifications: ui.notifications,
        projectId,
        result,
        revert: () => ui.settings.patchGenerateSettings(revertPatch, projectId),
        t,
      });
    }
  };

  const selectModel = (model: GenerateModelConfig) => {
    if (model.key === selectedModel?.key) {
      return;
    }

    const result = getGenerateModelSelectionResult({
      currentValues: draft.getSnapshot(),
      isCanvasFrame,
      model,
      models,
    });

    if (isLossyModelSwitch(result)) {
      requestGenerateModelSwitchConfirmation({ model, projectId });
      return;
    }

    commitModelSelection(model);
  };

  const focusModelSelect = () => document.getElementById(selectId)?.click();
  const hasNoSupportedModels = !isLoadingModels && !loadError && supportedModels.length === 0;

  return (
    <Stack gap="1" py="1">
      {/* The Field describes the picker with the model's issues, which have no one-step fix: the picker is it. */}
      <Field
        error={modelIssues.length > 0 ? modelIssues.map((issue) => describeGenerationIssue(t, issue)).join(' ') : null}
        hint="model"
        label={t('widgets.generate.model')}
      >
        <ModelSelect
          filter={(model) => isGenerateModelConfig(model) && isGenerateModelSelectable(model)}
          id={selectId}
          isClearable={false}
          modelTypes={MAIN_MODEL_TYPES}
          placeholder={
            isMissing && selectedModel
              ? t('widgets.generate.modelMissingPlaceholder', { name: selectedModel.name })
              : t('widgets.generate.selectModel')
          }
          value={selectedModel?.key ?? null}
          onChange={(model) => {
            if (isGenerateModelConfig(model) && isGenerateModelSelectable(model)) {
              selectModel(model);
            }
          }}
        />
      </Field>

      {isMissing ? (
        <HStack gap="1.5">
          <Button size="sm" variant="outline" onClick={focusModelSelect}>
            {t('widgets.generate.chooseAnotherModel')}
          </Button>
          <Button size="sm" variant="ghost" onClick={openManagerForMainModels}>
            {t('widgets.generate.openModelManager')}
          </Button>
        </HStack>
      ) : null}

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
          <Stack gap="1.5">
            <Text fontSize="lg">{t('widgets.generate.switchModelBody', { name: pendingSwitchModel?.name ?? '' })}</Text>
            <Stack as="ul" gap="1" listStyleType="disc" ps="5">
              {pendingLines.map((line) => (
                <Text key={line} as="li" color="fg.muted" fontSize="md">
                  {line}
                </Text>
              ))}
            </Stack>
          </Stack>
        }
        confirmLabel={t('widgets.generate.switchModelConfirm')}
        isOpen={pendingSwitchModel !== null}
        title={t('widgets.generate.switchModelTitle')}
        onClose={clearGenerateModelSwitchConfirmation}
        onConfirm={() => {
          if (pendingSwitchModel) {
            // Close before committing, so the dialog animates out listing what was confirmed, not the switched draft.
            clearGenerateModelSwitchConfirmation();
            commitModelSelection(pendingSwitchModel);
          }
        }}
      />
    </Stack>
  );
};
