/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings, Ideogram4SamplerPreset } from '@features/generation/core/types';

import {
  Badge,
  Box,
  createListCollection,
  HStack,
  Image,
  Input,
  InputGroup,
  NumberInput,
  Stack,
  Text,
} from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationModelPolicy } from '@features/generation/core/baseGenerationPolicies';
import {
  IDEOGRAM4_SAMPLER_PRESETS,
  MAX_KREA2_SEED_VARIANCE_STRENGTH,
  SEED_MAX,
} from '@features/generation/core/settings';
import { Combobox, Field, IconButton, Select, Tooltip } from '@platform/ui';
import { ModelDefaultButton } from '@platform/ui/ModelDefaultButton';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { DicesIcon, ShuffleIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { GenerateConditioningRebalanceField } from './GenerateConditioningRebalanceField';
import { useGenerationUi } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';
import { GenerateToggleSwitch } from './shared/GenerateToggleSwitch';

const STEPS_SLIDER_MAX = 100;

const SEED_END_ELEMENT_PROPS = { pointerEvents: 'auto', pr: '0.5' } as const;

interface GenerateRenderSectionProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

/**
 * Ideogram 4 presets fix a step count and guidance schedule. The labels carry the step count
 * because the backend's identifiers (`V4_QUALITY_48`) are otherwise opaque in a picker.
 */
const IDEOGRAM4_PRESET_LABELS: Record<Ideogram4SamplerPreset, string> = {
  V4_DEFAULT_20: 'Default (20 steps)',
  V4_QUALITY_48: 'Quality (48 steps)',
  V4_TURBO_12: 'Turbo (12 steps)',
};

const IDEOGRAM4_PRESET_COLLECTION = createListCollection({
  items: IDEOGRAM4_SAMPLER_PRESETS.map((value) => ({ label: IDEOGRAM4_PRESET_LABELS[value], value })),
});

/**
 * Family-specific sampling parameters, co-located with the shared sampling
 * controls they modify rather than segregated into a "model family" bucket.
 */
const Ideogram4SamplingFields = ({ onCommit, settings }: Pick<GenerateRenderSectionProps, 'onCommit' | 'settings'>) => {
  const { t } = useTranslation();

  return (
    <>
      <Field label={t('widgets.generate.ideogram4SamplerPreset')}>
        <Select
          aria-label={t('widgets.generate.ideogram4SamplerPreset')}
          collection={IDEOGRAM4_PRESET_COLLECTION}
          size="xs"
          value={[settings.ideogram4SamplerPreset]}
          onValueChange={({ value }) => {
            const preset = value[0];

            if (IDEOGRAM4_SAMPLER_PRESETS.includes(preset as Ideogram4SamplerPreset)) {
              onCommit({ ideogram4SamplerPreset: preset as Ideogram4SamplerPreset });
            }
          }}
        />
      </Field>
      {/*
        Steps, guidance and mu override the preset. Null means "let the preset decide", so each
        keeps its own enable switch rather than using a sentinel value in the slider's range.
      */}
      <Field label={t('widgets.generate.ideogram4Steps')} helpText={t('widgets.generate.ideogram4PresetDerived')}>
        <GenerateToggleSwitch
          checked={settings.ideogram4Steps !== null}
          label={t('widgets.generate.override')}
          labelVisible
          onCheckedChange={(checked) => onCommit({ ideogram4Steps: checked ? 48 : null })}
        />
        {settings.ideogram4Steps !== null ? (
          <SliderNumberField
            ariaLabel={t('widgets.generate.ideogram4Steps')}
            max={100}
            min={1}
            step={1}
            value={settings.ideogram4Steps}
            onChange={(value) => onCommit({ ideogram4Steps: value })}
          />
        ) : null}
      </Field>
      <Field
        label={t('widgets.generate.ideogram4GuidanceScale')}
        helpText={t('widgets.generate.ideogram4PresetDerived')}
      >
        <GenerateToggleSwitch
          checked={settings.ideogram4GuidanceScale !== null}
          label={t('widgets.generate.override')}
          labelVisible
          onCheckedChange={(checked) => onCommit({ ideogram4GuidanceScale: checked ? 5 : null })}
        />
        {settings.ideogram4GuidanceScale !== null ? (
          <SliderNumberField
            ariaLabel={t('widgets.generate.ideogram4GuidanceScale')}
            max={20}
            min={0}
            step={0.1}
            value={settings.ideogram4GuidanceScale}
            onChange={(value) => onCommit({ ideogram4GuidanceScale: value })}
          />
        ) : null}
      </Field>
      <Field label={t('widgets.generate.ideogram4Mu')} helpText={t('widgets.generate.ideogram4MuHelp')}>
        <GenerateToggleSwitch
          checked={settings.ideogram4Mu !== null}
          label={t('widgets.generate.override')}
          labelVisible
          onCheckedChange={(checked) => onCommit({ ideogram4Mu: checked ? 1 : null })}
        />
        {settings.ideogram4Mu !== null ? (
          <SliderNumberField
            ariaLabel={t('widgets.generate.ideogram4Mu')}
            max={10}
            min={0}
            step={0.1}
            value={settings.ideogram4Mu}
            onChange={(value) => onCommit({ ideogram4Mu: value })}
          />
        ) : null}
      </Field>
      <Field label={t('widgets.generate.ideogram4ColorPalette')} helpText={t('widgets.generate.ideogram4ColorHelp')}>
        <Input
          size="xs"
          value={settings.ideogram4ColorPalette.join(', ')}
          onChange={(event) =>
            onCommit({
              ideogram4ColorPalette: event.target.value
                .split(',')
                .map((entry) => entry.trim())
                .filter((entry) => entry !== ''),
            })
          }
        />
      </Field>
    </>
  );
};

/**
 * A14B runs two experts across the schedule. Null reuses the main guidance for the
 * low-noise half, which is the backend's own default.
 */
const WanLowNoiseGuidanceField = ({
  onCommit,
  settings,
}: Pick<GenerateRenderSectionProps, 'onCommit' | 'settings'>) => {
  const { t } = useTranslation();

  return (
    <Field label={t('widgets.generate.wanGuidanceLowNoise')} helpText={t('widgets.generate.wanGuidanceLowNoiseHelp')}>
      <GenerateToggleSwitch
        checked={settings.wanGuidanceScaleLowNoise !== null}
        label={t('widgets.generate.override')}
        labelVisible
        onCheckedChange={(checked) => onCommit({ wanGuidanceScaleLowNoise: checked ? settings.cfgScale : null })}
      />
      {settings.wanGuidanceScaleLowNoise !== null ? (
        <SliderNumberField
          ariaLabel={t('widgets.generate.wanGuidanceLowNoise')}
          max={20}
          min={1}
          step={0.1}
          value={settings.wanGuidanceScaleLowNoise}
          onChange={(value) => onCommit({ wanGuidanceScaleLowNoise: value })}
        />
      ) : null}
    </Field>
  );
};

/** Perturbs Krea-2 conditioning between seeds — a variation concern, so it sits by the seed. */
const Krea2SeedVarianceFields = ({ onCommit, settings }: Pick<GenerateRenderSectionProps, 'onCommit' | 'settings'>) => {
  const { t } = useTranslation();

  return (
    <>
      <Field label={t('widgets.generate.krea2SeedVariance')} helpText={t('widgets.generate.krea2SeedVarianceHelp')}>
        <GenerateToggleSwitch
          checked={settings.krea2SeedVarianceEnabled}
          label={t('widgets.generate.enabled')}
          labelVisible
          onCheckedChange={(checked) => onCommit({ krea2SeedVarianceEnabled: checked })}
        />
      </Field>
      {settings.krea2SeedVarianceEnabled ? (
        <>
          <Field label={t('widgets.generate.krea2SeedVarianceStrength')}>
            <SliderNumberField
              ariaLabel={t('widgets.generate.krea2SeedVarianceStrength')}
              max={MAX_KREA2_SEED_VARIANCE_STRENGTH}
              min={0}
              step={0.05}
              value={settings.krea2SeedVarianceStrength}
              onChange={(value) => onCommit({ krea2SeedVarianceStrength: value })}
            />
          </Field>
          <Field label={t('widgets.generate.krea2SeedVarianceRandomize')}>
            <SliderNumberField
              ariaLabel={t('widgets.generate.krea2SeedVarianceRandomize')}
              formatValue={(value) => `${value}%`}
              max={100}
              min={0}
              step={1}
              value={settings.krea2SeedVarianceRandomizePercent}
              onChange={(value) => onCommit({ krea2SeedVarianceRandomizePercent: value })}
            />
          </Field>
        </>
      ) : null}
    </>
  );
};

/**
 * One seed field with the shuffle-new-seed action inside it, and a random
 * toggle beside it: pressed means every queued run draws a fresh seed and the
 * pinned value goes quiet. `shouldRandomizeSeed` persists unchanged underneath.
 */
const SeedField = ({ onCommit, settings }: Pick<GenerateRenderSectionProps, 'onCommit' | 'settings'>) => {
  const { t } = useTranslation();
  const { seedHistory } = useGenerationUi().queueInsights;
  const randomLabel = t('widgets.generate.randomEachRun');

  return (
    <Field hint="seed" label={t('common.seed')}>
      <Stack gap="1" w="full">
        <HStack gap="1">
          <NumberInput.Root
            disabled={settings.shouldRandomizeSeed}
            max={SEED_MAX}
            min={0}
            size="xs"
            value={String(settings.seed)}
            w="full"
            onValueChange={({ valueAsNumber }) => {
              if (Number.isFinite(valueAsNumber)) {
                onCommit({ seed: valueAsNumber });
              }
            }}
          >
            <InputGroup
              endElement={
                <IconButton
                  aria-label={t('widgets.generate.newSeed')}
                  color="fg.muted"
                  disabled={settings.shouldRandomizeSeed}
                  size="2xs"
                  title={t('widgets.generate.newSeed')}
                  variant="ghost"
                  onClick={() => onCommit({ seed: Math.floor(Math.random() * SEED_MAX) })}
                >
                  <DicesIcon />
                </IconButton>
              }
              endElementProps={SEED_END_ELEMENT_PROPS}
            >
              <NumberInput.Input aria-label={t('common.seed')} />
            </InputGroup>
          </NumberInput.Root>
          <Tooltip content={randomLabel}>
            <IconButton
              aria-label={randomLabel}
              aria-pressed={settings.shouldRandomizeSeed}
              flexShrink="0"
              size="xs"
              variant={settings.shouldRandomizeSeed ? 'solid' : 'outline'}
              onClick={() => onCommit({ shouldRandomizeSeed: !settings.shouldRandomizeSeed })}
            >
              <ShuffleIcon />
            </IconButton>
          </Tooltip>
        </HStack>
        {/* The executed seeds of recent runs, each behind its result — a seed
            stops being a magic number and becomes "that image's recipe".
            Clicking one pins it, switching to fixed mode if needed. */}
        {seedHistory.length > 0 ? (
          <HStack gap="1" pt="0.5">
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.generate.recentSeeds')}
            </Text>
            {seedHistory.map((item) => (
              <Tooltip key={item.seed} content={t('widgets.generate.useSeed', { seed: item.seed })}>
                <Box
                  aria-label={t('widgets.generate.useSeed', { seed: item.seed })}
                  as="button"
                  bg="bg.emphasized"
                  borderColor={
                    !settings.shouldRandomizeSeed && settings.seed === item.seed ? 'accent.solid' : 'border.subtle'
                  }
                  borderWidth="1px"
                  boxSize="5"
                  cursor="pointer"
                  overflow="hidden"
                  rounded="3px"
                  onClick={() => onCommit({ seed: item.seed, shouldRandomizeSeed: false })}
                >
                  {item.thumbnailUrl ? (
                    <Image alt="" boxSize="full" draggable={false} objectFit="cover" src={item.thumbnailUrl} />
                  ) : null}
                </Box>
              </Tooltip>
            ))}
          </HStack>
        ) : null}
      </Stack>
    </Field>
  );
};

/**
 * Sampling and variation — split out of the model zone because how a model
 * samples is not which model it is. Family-specific parameters render beside
 * the shared control they modify.
 */
export const GenerateRenderSection = ({
  onCommit,
  onCommitImmediate,
  selectedModel,
  settings,
}: GenerateRenderSectionProps) => {
  const { t } = useTranslation();
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationModelPolicy(selectedModel, settings);
  const familyBase = selectedModel && selectedModel.type !== 'external_image_generator' ? selectedModel.base : null;

  const commitNumber = (key: 'cfgScale' | 'steps', value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: value });
  };

  const badges = (
    <>
      <Badge size="xs">
        {settings.steps} · {policy.ui.guidanceLabel} {settings.cfgScale}
      </Badge>
      {policy.ui.seedVisible ? (
        <Badge size="xs">{settings.shouldRandomizeSeed ? t('widgets.generate.random') : settings.seed}</Badge>
      ) : null}
    </>
  );

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.render')}
      defaultOpen={false}
      badges={badges}
      sectionId="render"
    >
      <Stack gap="2" p="2">
        <GenerateFieldContextMenu
          copyValue={() => String(settings.steps)}
          isAtDefault={modelDefaults !== null && settings.steps === modelDefaults.steps}
          onReset={modelDefaults ? () => onCommit({ steps: modelDefaults.steps }) : undefined}
        >
          <Field hint="steps" label={t('widgets.generate.steps')}>
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
        </GenerateFieldContextMenu>
        <GenerateFieldContextMenu
          copyValue={() => String(settings.cfgScale)}
          isAtDefault={modelDefaults !== null && settings.cfgScale === modelDefaults.cfgScale}
          onReset={modelDefaults ? () => onCommit({ cfgScale: modelDefaults.cfgScale }) : undefined}
        >
          <Field hint="guidance" label={policy.ui.guidanceLabel}>
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
        </GenerateFieldContextMenu>
        {familyBase === 'krea-2' ? (
          <GenerateConditioningRebalanceField
            settings={settings}
            onCommit={onCommit}
            onCommitImmediate={onCommitImmediate}
          />
        ) : null}
        {familyBase === 'wan' ? <WanLowNoiseGuidanceField settings={settings} onCommit={onCommit} /> : null}
        {policy.ui.schedulerVisible ? (
          <GenerateFieldContextMenu
            copyValue={() => settings.scheduler}
            isAtDefault={modelDefaults !== null && settings.scheduler === modelDefaults.scheduler}
            onReset={modelDefaults ? () => onCommit({ scheduler: modelDefaults.scheduler }) : undefined}
          >
            <Field hint="scheduler" label={t('widgets.generate.scheduler')}>
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
          </GenerateFieldContextMenu>
        ) : null}
        {familyBase === 'ideogram-4' ? <Ideogram4SamplingFields settings={settings} onCommit={onCommit} /> : null}
        {policy.ui.seedVisible ? <SeedField settings={settings} onCommit={onCommit} /> : null}
        {familyBase === 'krea-2' ? <Krea2SeedVarianceFields settings={settings} onCommit={onCommit} /> : null}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
