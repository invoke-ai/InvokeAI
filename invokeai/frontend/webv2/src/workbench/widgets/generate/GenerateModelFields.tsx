/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type { ChangeEvent } from 'react';

import { HStack, Icon, InputGroup, NativeSelect, NumberInput, Stack } from '@chakra-ui/react';
import { Button, IconButton, Field, Tooltip } from '@workbench/components/ui';
import {
  getDefaultGenerateSettings,
  getGenerationModelPolicy,
  getSettingsWithCompatibleModelSelections,
  getSettingsWithModelDefaults,
  isGenerateModelSelectable,
} from '@workbench/generation/baseGenerationPolicies';
import {
  getModelDefaultVae,
  hasModelDefaultVae,
  isGenerateModelConfig,
  SEED_MAX,
} from '@workbench/generation/settings';
import { ModelSelect } from '@workbench/models/components';
import { useNotify } from '@workbench/useNotify';
import { ArrowRightLeft, DicesIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';

const formatClearedSettings = (labels: readonly string[]) =>
  labels.length <= 1 ? labels[0] : `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]}`;

interface GenerateModelFieldsProps {
  settings: GenerateSettings;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  vaeModels: VaeModelConfig[];
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

const MODEL_DEFAULT_VALUE_KEYS = [
  'aspectRatioId',
  'aspectRatioIsLocked',
  'aspectRatioValue',
  'cfgRescaleMultiplier',
  'cfgScale',
  'height',
  'modelKey',
  'scheduler',
  'steps',
  'vaePrecision',
  'width',
] as const satisfies readonly (keyof GenerateSettings)[];

const settingsMatchModelDefaults = (settings: GenerateSettings, modelDefaultSettings: GenerateSettings) =>
  MODEL_DEFAULT_VALUE_KEYS.every((key) => Object.is(settings[key], modelDefaultSettings[key])) &&
  settings.vae?.key === modelDefaultSettings.vae?.key &&
  settings.loras.length === modelDefaultSettings.loras.length &&
  settings.loras.every((lora, index) => lora.isEnabled === modelDefaultSettings.loras[index]?.isEnabled);

export const GenerateModelFields = ({
  models,
  onCommit,
  onCommitSettings,
  selectedModel,
  settings,
  vaeModels,
}: GenerateModelFieldsProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationModelPolicy(selectedModel, settings);

  const getModelDefaultSettings = (model: GenerateModelConfig) => {
    const nextSettings = getSettingsWithModelDefaults(settings, model);

    return hasModelDefaultVae(model) ? { ...nextSettings, vae: getModelDefaultVae(model, vaeModels) } : nextSettings;
  };

  const modelDefaultSettings = selectedModel ? getModelDefaultSettings(selectedModel) : null;
  const modelDefaultsAreApplied = modelDefaultSettings
    ? settingsMatchModelDefaults(settings, modelDefaultSettings)
    : false;

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
      notify.info(
        t('widgets.generate.incompatibleSettingsCleared'),
        t('widgets.generate.incompatibleSettingsClearedDescription', {
          count: result.clearedLabels.length,
          labels: formatClearedSettings(result.clearedLabels),
          name: model.name,
        })
      );
    }
  };

  const applyModelDefaults = (model: GenerateModelConfig) => {
    onCommitSettings(getModelDefaultSettings(model));
  };

  return (
    <GenerateCollapsibleSection label={t('widgets.generate.model')} defaultOpen>
      <Stack gap="2" p="2">
        <Field label={t('widgets.generate.model')}>
          <HStack gap="1">
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
            <ModelDefaultButton
              active={Boolean(modelDefaultSettings && !modelDefaultsAreApplied)}
              disabled={!selectedModel}
              onClick={() => {
                if (selectedModel) {
                  applyModelDefaults(selectedModel);
                }
              }}
            />
          </HStack>
        </Field>
        <HStack gap="2">
          <Field label={t('widgets.generate.steps')}>
            <NumberInput.Root
              size="xs"
              value={String(settings.steps)}
              onValueChange={updateNumberInput('steps')}
              min={1}
              allowMouseWheel
            >
              <InputGroup
                startElementProps={{ pointerEvents: 'auto' }}
                startElement={
                  <NumberInput.Scrubber>
                    <Icon as={ArrowRightLeft} boxSize="3" />
                  </NumberInput.Scrubber>
                }
                endElement={
                  <ModelDefaultButton
                    disabled={!modelDefaults || settings.steps === modelDefaults.steps}
                    label={t('widgets.generate.useModelDefaultSteps')}
                    onClick={() => {
                      if (modelDefaults) {
                        onCommit({ steps: modelDefaults.steps });
                      }
                    }}
                  />
                }
                endElementProps={{ pointerEvents: 'auto' }}
              >
                <NumberInput.Input />
              </InputGroup>
            </NumberInput.Root>
          </Field>
          <Field label={policy.ui.guidanceLabel}>
            <NumberInput.Root
              size="xs"
              value={String(settings.cfgScale)}
              onValueChange={updateNumberInput('cfgScale')}
              min={0}
              max={10}
              step={0.5}
            >
              <InputGroup
                endElement={
                  <ModelDefaultButton
                    disabled={!modelDefaults || settings.cfgScale === modelDefaults.cfgScale}
                    label={t('widgets.generate.useModelDefaultField', { field: policy.ui.guidanceLabel })}
                    onClick={() => {
                      if (modelDefaults) {
                        onCommit({ cfgScale: modelDefaults.cfgScale });
                      }
                    }}
                  />
                }
                endElementProps={{ pointerEvents: 'auto' }}
              >
                <NumberInput.Input />
              </InputGroup>
            </NumberInput.Root>
          </Field>
          {policy.ui.schedulerVisible ? (
            <Field label={t('widgets.generate.scheduler')}>
              <HStack gap="1">
                <NativeSelect.Root flex="1" size="xs">
                  <NativeSelect.Field
                    aria-label={t('widgets.generate.scheduler')}
                    value={settings.scheduler}
                    onChange={(event: ChangeEvent<HTMLSelectElement>) =>
                      onCommit({ scheduler: event.currentTarget.value })
                    }
                  >
                    {policy.scheduler.options.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </NativeSelect.Field>
                  <NativeSelect.Indicator />
                </NativeSelect.Root>
                <ModelDefaultButton
                  disabled={!modelDefaults || settings.scheduler === modelDefaults.scheduler}
                  label={t('widgets.generate.useModelDefaultScheduler')}
                  onClick={() => {
                    if (modelDefaults) {
                      onCommit({ scheduler: modelDefaults.scheduler });
                    }
                  }}
                />
              </HStack>
            </Field>
          ) : null}
        </HStack>
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
                  disabled={settings.shouldRandomizeSeed}
                  size="xs"
                  variant="outline"
                  onClick={() => onCommit({ seed: Math.floor(Math.random() * SEED_MAX) })}
                >
                  <DicesIcon />
                </IconButton>
              </Tooltip>
              <Button
                size="xs"
                variant={settings.shouldRandomizeSeed ? 'solid' : 'outline'}
                onClick={() => onCommit({ shouldRandomizeSeed: !settings.shouldRandomizeSeed })}
              >
                {t('widgets.generate.random')}
              </Button>
            </HStack>
          </Field>
        ) : null}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
