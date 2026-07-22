import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { HStack, NumberInput, Stack, Switch } from '@chakra-ui/react';
import {
  getDefaultGenerateSettings,
  getGenerationModelPolicy,
  getSettingsWithCompatibleModelSelections,
  isGenerateModelSelectable,
} from '@features/generation/core/baseGenerationPolicies';
import { isGenerateModelConfig, SEED_MAX } from '@features/generation/core/settings';
import { Combobox, IconButton, Field, Tooltip } from '@platform/ui';
import { DicesIcon } from 'lucide-react';
import { useId } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';
import { SliderNumberField } from './shared/SliderNumberField';

const formatClearedSettings = (labels: readonly string[]) =>
  labels.length <= 1 ? labels[0] : `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]}`;

const STEPS_SLIDER_MAX = 100;

interface GenerateModelFieldsProps {
  settings: GenerateSettings;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

export const GenerateModelFields = ({
  models,
  onCommit,
  onCommitSettings,
  selectedModel,
  settings,
}: GenerateModelFieldsProps) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  // The switch shares the seed Field.Root with the number input; an explicit id
  // keeps its label bound to its own hidden input instead of the Field's control id.
  const randomSeedSwitchInputId = useId();
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationModelPolicy(selectedModel, settings);

  const commitNumber = (key: 'cfgScale' | 'seed' | 'steps', value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: value });
  };

  const updateNumberInput =
    (key: 'cfgScale' | 'seed' | 'steps') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      commitNumber(key, valueAsNumber);
    };

  const selectModel = (model: GenerateModelConfig) => {
    const result = getSettingsWithCompatibleModelSelections(settings, model, models);

    onCommitSettings(result.settings);

    if (result.clearedLabels.length > 0) {
      notifications.info(
        t('widgets.generate.incompatibleSettingsCleared'),
        t('widgets.generate.incompatibleSettingsClearedDescription', {
          count: result.clearedLabels.length,
          labels: formatClearedSettings(result.clearedLabels),
          name: model.name,
        })
      );
    }
  };

  return (
    <GenerateCollapsibleSection label={t('widgets.generate.model')} defaultOpen sectionId="model">
      <Stack gap="2" p="2">
        <Field label={t('widgets.generate.model')}>
          <ModelSelect
            filter={(model) => isGenerateModelConfig(model) && isGenerateModelSelectable(model)}
            isClearable={false}
            modelTypes={['main', 'external_image_generator']}
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
        <Field label={t('widgets.generate.steps')}>
          <SliderNumberField
            ariaLabel={t('widgets.generate.steps')}
            defaultValue={modelDefaults?.steps}
            marks={modelDefaults ? [modelDefaults.steps] : undefined}
            max={STEPS_SLIDER_MAX}
            min={1}
            numberInputMax={Number.MAX_SAFE_INTEGER}
            resetLabel={t('widgets.generate.useModelDefaultSteps')}
            step={1}
            value={settings.steps}
            onChange={(steps) => commitNumber('steps', steps)}
          />
        </Field>
        <Field label={policy.ui.guidanceLabel}>
          <SliderNumberField
            ariaLabel={policy.ui.guidanceLabel}
            defaultValue={modelDefaults?.cfgScale}
            marks={modelDefaults ? [modelDefaults.cfgScale] : undefined}
            max={10}
            min={0}
            resetLabel={t('widgets.generate.useModelDefaultField', { field: policy.ui.guidanceLabel })}
            step={0.5}
            value={settings.cfgScale}
            onChange={(cfgScale) => commitNumber('cfgScale', cfgScale)}
          />
        </Field>
        {policy.ui.schedulerVisible ? (
          <Field label={t('widgets.generate.scheduler')}>
            <HStack gap="1">
              <Combobox
                aria-label={t('widgets.generate.scheduler')}
                flex="1"
                options={policy.scheduler.options}
                size="xs"
                value={settings.scheduler}
                onValueChange={(scheduler) => onCommit({ scheduler })}
              />
              {modelDefaults && settings.scheduler !== modelDefaults.scheduler ? (
                <ModelDefaultButton
                  label={t('widgets.generate.useModelDefaultScheduler')}
                  onClick={() => onCommit({ scheduler: modelDefaults.scheduler })}
                />
              ) : null}
            </HStack>
          </Field>
        ) : null}
        {policy.ui.seedVisible ? (
          <Field label={t('common.seed')}>
            <HStack gap="2">
              <NumberInput.Root
                size="xs"
                value={String(settings.seed)}
                onValueChange={updateNumberInput('seed')}
                min={0}
                max={SEED_MAX}
                w="full"
                disabled={settings.shouldRandomizeSeed}
              >
                <NumberInput.Input />
              </NumberInput.Root>
              <Tooltip content={t('widgets.generate.shuffleSeed')}>
                <IconButton
                  aria-label={t('widgets.generate.shuffleSeed')}
                  color="fg.muted"
                  disabled={settings.shouldRandomizeSeed}
                  size="xs"
                  variant="ghost"
                  onClick={() => onCommit({ seed: Math.floor(Math.random() * SEED_MAX) })}
                >
                  <DicesIcon />
                </IconButton>
              </Tooltip>
              <Switch.Root
                checked={settings.shouldRandomizeSeed}
                flexShrink="0"
                ids={{ hiddenInput: randomSeedSwitchInputId, label: `${randomSeedSwitchInputId}-label` }}
                size="sm"
                onCheckedChange={(event) => onCommit({ shouldRandomizeSeed: event.checked })}
              >
                <Switch.HiddenInput />
                <Switch.Control _checked={{ bg: 'accent.solid' }}>
                  <Switch.Thumb />
                </Switch.Control>
                <Switch.Label fontSize="xs">{t('widgets.generate.random')}</Switch.Label>
              </Switch.Root>
            </HStack>
          </Field>
        ) : null}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
