/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, Stack } from '@chakra-ui/react';
import { getGenerationUiPolicy, getSchedulerOptions } from '@features/generation/core/baseGenerationPolicies';
import {
  DEFAULT_REFINER_CFG_SCALE,
  DEFAULT_REFINER_NEGATIVE_AESTHETIC_SCORE,
  DEFAULT_REFINER_POSITIVE_AESTHETIC_SCORE,
  DEFAULT_REFINER_SCHEDULER,
  DEFAULT_REFINER_START,
  DEFAULT_REFINER_STEPS,
  isRefinerModel,
  MAX_REFINER_AESTHETIC_SCORE,
  MAX_REFINER_START,
  MIN_REFINER_AESTHETIC_SCORE,
  MIN_REFINER_CFG_SCALE,
  MIN_REFINER_START,
  MIN_REFINER_STEPS,
} from '@features/generation/core/sdxlRefiner';
import { isMainModelConfig } from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Combobox } from '@platform/ui/Combobox';
import { Field } from '@platform/ui/Field';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import { GenerationModelSelect as ModelSelect } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';

/** Practical track ranges; typed values may exceed them within validation bounds. */
const STEPS_SLIDER_MAX = 100;
const CFG_SLIDER_MAX = 20;
const CFG_INPUT_MAX = 100;

interface GenerateRefinerSectionProps {
  draft: GenerateDraft;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const selectRefinerSettings = pickGenerateSettings([
  'refinerCfgScale',
  'refinerModel',
  'refinerNegativeAestheticScore',
  'refinerPositiveAestheticScore',
  'refinerScheduler',
  'refinerStart',
  'refinerSteps',
]);

type RefinerNumberKey =
  | 'refinerCfgScale'
  | 'refinerNegativeAestheticScore'
  | 'refinerPositiveAestheticScore'
  | 'refinerStart'
  | 'refinerSteps';

const refinerFilter = (model: ModelConfig) => isRefinerModel(model);
const MAIN_MODEL_TYPES = ['main'];

export const GenerateRefinerSection = ({
  draft,
  models,
  onCommit,
  onCommitImmediate,
  selectedModel,
}: GenerateRefinerSectionProps) => {
  const { t } = useTranslation();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectRefinerSettings);
  const commitRefinerModel = useCallback(
    (model: ModelConfig | null) => {
      onCommitImmediate({ refinerModel: isMainModelConfig(model) && isRefinerModel(model) ? model : null });
    },
    [onCommitImmediate]
  );

  if (!getGenerationUiPolicy(selectedModel).refinerVisible) {
    return null;
  }

  const hasInstalledRefiner = models.some(refinerFilter);
  const refiner = settings.refinerModel;
  const isActive = refiner !== null;

  const commitNumber = (key: RefinerNumberKey, min: number, max: number) => (value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: Math.min(max, Math.max(min, value)) });
  };

  const numberField = ({
    defaultValue,
    hint,
    inputMax,
    key,
    label,
    max,
    min,
    step,
  }: {
    defaultValue: number;
    hint: 'refinerSteps' | 'refinerCfgScale' | 'refinerStart' | 'refinerAestheticScore';
    inputMax?: number;
    key: RefinerNumberKey;
    label: string;
    max: number;
    min: number;
    step: number;
  }) => (
    <GenerateFieldContextMenu
      copyValue={() => String(settings[key])}
      isAtDefault={!isActive || settings[key] === defaultValue}
      onReset={() => onCommit({ [key]: defaultValue })}
    >
      <ScrubberField
        defaultValue={defaultValue}
        disabled={!isActive}
        hint={hint}
        inputMax={inputMax}
        label={label}
        max={max}
        min={min}
        step={step}
        value={settings[key]}
        onChange={commitNumber(key, min, inputMax ?? max)}
      />
    </GenerateFieldContextMenu>
  );

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.refiner')}
      defaultOpen={false}
      badges={isActive ? <Badge>{refiner.name}</Badge> : null}
      sectionId="refiner"
    >
      <Stack gap="2" p="2">
        <Field
          hint="refinerModel"
          label={t('widgets.generate.refinerModel')}
          helpText={hasInstalledRefiner ? undefined : t('widgets.generate.noRefinerModels')}
        >
          <ModelSelect
            filter={refinerFilter}
            isClearable
            modelTypes={MAIN_MODEL_TYPES}
            placeholder={t('widgets.generate.refinerNone')}
            value={refiner?.key ?? null}
            onChange={commitRefinerModel}
          />
        </Field>
        {numberField({
          defaultValue: DEFAULT_REFINER_STEPS,
          hint: 'refinerSteps',
          inputMax: Number.MAX_SAFE_INTEGER,
          key: 'refinerSteps',
          label: t('widgets.generate.refinerSteps'),
          max: STEPS_SLIDER_MAX,
          min: MIN_REFINER_STEPS,
          step: 1,
        })}
        {numberField({
          defaultValue: DEFAULT_REFINER_CFG_SCALE,
          hint: 'refinerCfgScale',
          inputMax: CFG_INPUT_MAX,
          key: 'refinerCfgScale',
          label: t('widgets.generate.refinerCfgScale'),
          max: CFG_SLIDER_MAX,
          min: MIN_REFINER_CFG_SCALE,
          step: 0.5,
        })}
        <GenerateFieldContextMenu
          copyValue={() => settings.refinerScheduler}
          isAtDefault={!isActive || settings.refinerScheduler === DEFAULT_REFINER_SCHEDULER}
          onReset={() => onCommitImmediate({ refinerScheduler: DEFAULT_REFINER_SCHEDULER })}
        >
          <Field disabled={!isActive} hint="refinerScheduler" label={t('widgets.generate.refinerScheduler')}>
            <Combobox
              aria-label={t('widgets.generate.refinerScheduler')}
              disabled={!isActive}
              options={getSchedulerOptions(selectedModel, settings.refinerScheduler)}
              value={settings.refinerScheduler}
              onValueChange={(refinerScheduler) => onCommitImmediate({ refinerScheduler })}
            />
          </Field>
        </GenerateFieldContextMenu>
        {numberField({
          defaultValue: DEFAULT_REFINER_START,
          hint: 'refinerStart',
          key: 'refinerStart',
          label: t('widgets.generate.refinerStart'),
          max: MAX_REFINER_START,
          min: MIN_REFINER_START,
          step: 0.01,
        })}
        {numberField({
          defaultValue: DEFAULT_REFINER_POSITIVE_AESTHETIC_SCORE,
          hint: 'refinerAestheticScore',
          key: 'refinerPositiveAestheticScore',
          label: t('widgets.generate.refinerPositiveAestheticScore'),
          max: MAX_REFINER_AESTHETIC_SCORE,
          min: MIN_REFINER_AESTHETIC_SCORE,
          step: 0.5,
        })}
        {numberField({
          defaultValue: DEFAULT_REFINER_NEGATIVE_AESTHETIC_SCORE,
          hint: 'refinerAestheticScore',
          key: 'refinerNegativeAestheticScore',
          label: t('widgets.generate.refinerNegativeAestheticScore'),
          max: MAX_REFINER_AESTHETIC_SCORE,
          min: MIN_REFINER_AESTHETIC_SCORE,
          step: 0.5,
        })}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
