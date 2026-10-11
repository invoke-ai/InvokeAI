/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, createListCollection, Stack } from '@chakra-ui/react';
import {
  DEFAULT_GEMINI_TEMPERATURE,
  GEMINI_TEMPERATURE_MAX,
  GEMINI_TEMPERATURE_MIN,
  GEMINI_THINKING_LEVELS,
  getExternalProviderOptionKeys,
  isGeminiThinkingLevel,
  isOpenAIImageBackground,
  isOpenAIImageQuality,
  isOpenAIInputFidelity,
  OPENAI_IMAGE_BACKGROUNDS,
  OPENAI_IMAGE_QUALITIES,
  OPENAI_INPUT_FIDELITIES,
} from '@features/generation/core/externalProvider';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Field } from '@platform/ui/Field';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Select } from '@platform/ui/Select';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateToggleSwitch } from './shared/GenerateToggleSwitch';

interface GenerateProviderOptionsSectionProps {
  draft: GenerateDraft;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const selectProviderSettings = pickGenerateSettings([
  'geminiTemperature',
  'geminiThinkingLevel',
  'openaiBackground',
  'openaiInputFidelity',
  'openaiQuality',
  'seedreamOptimizePrompt',
  'seedreamWatermark',
]);

/** The empty value stands for "let the provider decide" on the optional selects. */
const PROVIDER_DEFAULT = '';

/** Options a hosted provider's node accepts beyond the shared prompt/size/seed fields; hidden for local models. */
export const GenerateProviderOptionsSection = ({
  draft,
  onCommit,
  selectedModel,
}: GenerateProviderOptionsSectionProps) => {
  const { t } = useTranslation();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectProviderSettings);
  const optionKeys = getExternalProviderOptionKeys(selectedModel);
  const collections = useMemo(
    () => ({
      background: createListCollection({
        items: OPENAI_IMAGE_BACKGROUNDS.map((value) => ({
          label: t(`widgets.generate.providerOptions.openaiBackgroundOptions.${value}`),
          value,
        })),
      }),
      inputFidelity: createListCollection({
        items: [
          { label: t('widgets.generate.providerOptions.providerDefault'), value: PROVIDER_DEFAULT },
          ...OPENAI_INPUT_FIDELITIES.map((value) => ({
            label: t(`widgets.generate.providerOptions.openaiInputFidelityOptions.${value}`),
            value,
          })),
        ],
      }),
      quality: createListCollection({
        items: OPENAI_IMAGE_QUALITIES.map((value) => ({
          label: t(`widgets.generate.providerOptions.openaiQualityOptions.${value}`),
          value,
        })),
      }),
      thinkingLevel: createListCollection({
        items: [
          { label: t('widgets.generate.providerOptions.providerDefault'), value: PROVIDER_DEFAULT },
          ...GEMINI_THINKING_LEVELS.map((value) => ({
            label: t(`widgets.generate.providerOptions.geminiThinkingLevelOptions.${value}`),
            value,
          })),
        ],
      }),
    }),
    [t]
  );

  if (optionKeys.length === 0) {
    return null;
  }

  // Count decisions away from the provider defaults, as the Advanced section does.
  const overrideCount = optionKeys.filter((key) => {
    switch (key) {
      case 'openaiQuality':
      case 'openaiBackground':
        return settings[key] !== 'auto';
      case 'seedreamWatermark':
      case 'seedreamOptimizePrompt':
        return settings[key];
      default:
        return settings[key] !== null;
    }
  }).length;

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.providerOptions.title')}
      defaultOpen
      badges={
        overrideCount > 0 ? <Badge>{t('widgets.generate.overridesCount', { count: overrideCount })}</Badge> : null
      }
      sectionId="providerOptions"
    >
      <Stack gap="2" p="2">
        {optionKeys.includes('openaiQuality') ? (
          <Field
            label={t('widgets.generate.providerOptions.openaiQuality')}
            helpText={t('widgets.generate.providerOptions.openaiQualityHelp')}
          >
            <Select
              aria-label={t('widgets.generate.providerOptions.openaiQuality')}
              collection={collections.quality}
              value={[settings.openaiQuality]}
              onValueChange={({ value }) => {
                if (isOpenAIImageQuality(value[0])) {
                  onCommit({ openaiQuality: value[0] });
                }
              }}
            />
          </Field>
        ) : null}
        {optionKeys.includes('openaiBackground') ? (
          <Field
            label={t('widgets.generate.providerOptions.openaiBackground')}
            helpText={t('widgets.generate.providerOptions.openaiBackgroundHelp')}
          >
            <Select
              aria-label={t('widgets.generate.providerOptions.openaiBackground')}
              collection={collections.background}
              value={[settings.openaiBackground]}
              onValueChange={({ value }) => {
                if (isOpenAIImageBackground(value[0])) {
                  onCommit({ openaiBackground: value[0] });
                }
              }}
            />
          </Field>
        ) : null}
        {optionKeys.includes('openaiInputFidelity') ? (
          <Field
            label={t('widgets.generate.providerOptions.openaiInputFidelity')}
            helpText={t('widgets.generate.providerOptions.openaiInputFidelityHelp')}
          >
            <Select
              aria-label={t('widgets.generate.providerOptions.openaiInputFidelity')}
              collection={collections.inputFidelity}
              value={[settings.openaiInputFidelity ?? PROVIDER_DEFAULT]}
              onValueChange={({ value }) => {
                const fidelity = value[0];

                if (fidelity === PROVIDER_DEFAULT) {
                  onCommit({ openaiInputFidelity: null });
                } else if (isOpenAIInputFidelity(fidelity)) {
                  onCommit({ openaiInputFidelity: fidelity });
                }
              }}
            />
          </Field>
        ) : null}
        {optionKeys.includes('geminiTemperature') ? (
          <Field
            label={t('widgets.generate.providerOptions.geminiTemperature')}
            helpText={t('widgets.generate.providerOptions.geminiTemperatureHelp')}
            labelEnd={
              <GenerateToggleSwitch
                checked={settings.geminiTemperature !== null}
                label={t('widgets.generate.override')}
                labelVisible
                onCheckedChange={(checked) =>
                  onCommit({ geminiTemperature: checked ? DEFAULT_GEMINI_TEMPERATURE : null })
                }
              />
            }
          >
            {settings.geminiTemperature === null ? null : (
              <ScrubberField
                defaultValue={DEFAULT_GEMINI_TEMPERATURE}
                label={t('widgets.generate.providerOptions.geminiTemperature')}
                marks={[DEFAULT_GEMINI_TEMPERATURE]}
                max={GEMINI_TEMPERATURE_MAX}
                min={GEMINI_TEMPERATURE_MIN}
                step={0.05}
                value={settings.geminiTemperature}
                onChange={(value) => {
                  if (Number.isFinite(value)) {
                    onCommit({
                      geminiTemperature: Math.min(GEMINI_TEMPERATURE_MAX, Math.max(GEMINI_TEMPERATURE_MIN, value)),
                    });
                  }
                }}
              />
            )}
          </Field>
        ) : null}
        {optionKeys.includes('geminiThinkingLevel') ? (
          <Field
            label={t('widgets.generate.providerOptions.geminiThinkingLevel')}
            helpText={t('widgets.generate.providerOptions.geminiThinkingLevelHelp')}
          >
            <Select
              aria-label={t('widgets.generate.providerOptions.geminiThinkingLevel')}
              collection={collections.thinkingLevel}
              value={[settings.geminiThinkingLevel ?? PROVIDER_DEFAULT]}
              onValueChange={({ value }) => {
                const level = value[0];

                if (level === PROVIDER_DEFAULT) {
                  onCommit({ geminiThinkingLevel: null });
                } else if (isGeminiThinkingLevel(level)) {
                  onCommit({ geminiThinkingLevel: level });
                }
              }}
            />
          </Field>
        ) : null}
        {optionKeys.includes('seedreamWatermark') ? (
          <Field
            label={t('widgets.generate.providerOptions.seedreamWatermark')}
            helpText={t('widgets.generate.providerOptions.seedreamWatermarkHelp')}
          >
            <GenerateToggleSwitch
              checked={settings.seedreamWatermark}
              label={t('widgets.generate.providerOptions.seedreamWatermark')}
              onCheckedChange={(checked) => onCommit({ seedreamWatermark: checked })}
            />
          </Field>
        ) : null}
        {optionKeys.includes('seedreamOptimizePrompt') ? (
          <Field
            label={t('widgets.generate.providerOptions.seedreamOptimizePrompt')}
            helpText={t('widgets.generate.providerOptions.seedreamOptimizePromptHelp')}
          >
            <GenerateToggleSwitch
              checked={settings.seedreamOptimizePrompt}
              label={t('widgets.generate.providerOptions.seedreamOptimizePrompt')}
              onCheckedChange={(checked) => onCommit({ seedreamOptimizePrompt: checked })}
            />
          </Field>
        ) : null}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
