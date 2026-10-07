/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings, Ideogram4SamplerPreset } from '@features/generation/core/types';
import type { ReactNode } from 'react';

import { Badge, Box, createListCollection, HStack, Image, Input, Separator, Stack, Text } from '@chakra-ui/react';
import {
  getDefaultGenerateSettings,
  getGenerationUiPolicy,
  getGuidanceBoundReason,
  getSchedulerOptions,
} from '@features/generation/core/baseGenerationPolicies';
import { getEffectivePrompts } from '@features/generation/core/promptTemplates';
import {
  getDynamicPromptsConfig,
  IDEOGRAM4_GUIDANCE_MAX,
  IDEOGRAM4_GUIDANCE_MIN,
  IDEOGRAM4_MU_MAX,
  IDEOGRAM4_MU_MIN,
  IDEOGRAM4_SAMPLER_PRESETS,
  IDEOGRAM4_STEPS_MAX,
  IDEOGRAM4_STEPS_MIN,
  MAX_KREA2_SEED_VARIANCE_STRENGTH,
} from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Combobox } from '@platform/ui/Combobox';
import { Field } from '@platform/ui/Field';
import { ModelDefaultButton } from '@platform/ui/ModelDefaultButton';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Select } from '@platform/ui/Select';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { Tooltip } from '@platform/ui/Tooltip';
import { useTranslation } from 'react-i18next';

import { GenerateConditioningRebalanceField } from './GenerateConditioningRebalanceField';
import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import {
  type CanvasDenoisingStrength,
  type GenerationQueueInsights,
  useGenerationQueueInsights,
  useGenerationUi,
} from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';
import { GenerateToggleSwitch } from './shared/GenerateToggleSwitch';
import { SeedField as SharedSeedField } from './shared/SeedField';
import { useDynamicPrompts } from './useDynamicPrompts';

const STEPS_SLIDER_MAX = 100;
const formatPercent = (value: number): string => `${value}%`;

/**
 * Slider tracks are practical ranges, distinct from input/validation bounds; FLUX Fill's default 30 exceeds its
 * track.
 */
const GUIDANCE_SLIDER_MAX = 10;
const GUIDANCE_INPUT_MAX = 100;

interface GenerateRenderSectionProps {
  draft: GenerateDraft;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const selectRenderSettings = pickGenerateSettings([
  'batchCount',
  'cfgScale',
  'dynamicPromptsCombinatorial',
  'dynamicPromptsMaxPrompts',
  'dynamicPromptsResample',
  'dynamicPromptsSampleSeed',
  'dynamicPromptsSeedBehaviour',
  'ideogram4ColorPalette',
  'ideogram4GuidanceScale',
  'ideogram4Mu',
  'ideogram4SamplerPreset',
  'ideogram4Steps',
  'krea2RebalanceEnabled',
  'krea2RebalanceMultiplier',
  'krea2RebalanceWeights',
  'krea2SeedVarianceEnabled',
  'krea2SeedVarianceRandomizePercent',
  'krea2SeedVarianceStrength',
  // The seed field counts the prompts the effective positive prompt expands to.
  'negativePrompt',
  'positivePrompt',
  'promptTemplate',
  'scheduler',
  'seed',
  'seedMode',
  'steps',
  'wanGuidanceScaleLowNoise',
]);

interface RenderFieldsProps {
  settings: ReturnType<typeof selectRenderSettings>;
  onCommit: GenerateRenderSectionProps['onCommit'];
}

/** Labels expose step counts hidden in backend preset IDs. */
const IDEOGRAM4_PRESET_LABELS: Record<Ideogram4SamplerPreset, string> = {
  V4_DEFAULT_20: 'Default (20 steps)',
  V4_QUALITY_48: 'Quality (48 steps)',
  V4_TURBO_12: 'Turbo (12 steps)',
};

const IDEOGRAM4_PRESET_COLLECTION = createListCollection({
  items: IDEOGRAM4_SAMPLER_PRESETS.map((value) => ({ label: IDEOGRAM4_PRESET_LABELS[value], value })),
});

const Ideogram4SamplingFields = ({ onCommit, settings }: RenderFieldsProps) => {
  const { t } = useTranslation();

  return (
    <>
      <Field label={t('widgets.generate.ideogram4SamplerPreset')}>
        <Select
          aria-label={t('widgets.generate.ideogram4SamplerPreset')}
          collection={IDEOGRAM4_PRESET_COLLECTION}
          value={[settings.ideogram4SamplerPreset]}
          onValueChange={({ value }) => {
            const preset = value[0];

            if (IDEOGRAM4_SAMPLER_PRESETS.includes(preset as Ideogram4SamplerPreset)) {
              onCommit({ ideogram4SamplerPreset: preset as Ideogram4SamplerPreset });
            }
          }}
        />
      </Field>
      {/* Null inherits the preset; switches avoid numeric sentinels. */}
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
            max={IDEOGRAM4_STEPS_MAX}
            min={IDEOGRAM4_STEPS_MIN}
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
            max={IDEOGRAM4_GUIDANCE_MAX}
            min={IDEOGRAM4_GUIDANCE_MIN}
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
            max={IDEOGRAM4_MU_MAX}
            min={IDEOGRAM4_MU_MIN}
            step={0.1}
            value={settings.ideogram4Mu}
            onChange={(value) => onCommit({ ideogram4Mu: value })}
          />
        ) : null}
      </Field>
      <Field label={t('widgets.generate.ideogram4ColorPalette')} helpText={t('widgets.generate.ideogram4ColorHelp')}>
        <Input
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

/** Null low-noise guidance inherits main guidance. */
const WanLowNoiseGuidanceField = ({ onCommit, settings }: RenderFieldsProps) => {
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
const Krea2SeedVarianceFields = ({ onCommit, settings }: RenderFieldsProps) => {
  const { t } = useTranslation();

  return (
    <>
      <Field
        helpText={t('widgets.generate.krea2SeedVarianceHelp')}
        label={t('widgets.generate.krea2SeedVariance')}
        labelEnd={
          <GenerateToggleSwitch
            checked={settings.krea2SeedVarianceEnabled}
            label={t('widgets.generate.krea2SeedVariance')}
            onCheckedChange={(checked) => onCommit({ krea2SeedVarianceEnabled: checked })}
          />
        }
      >
        {null}
      </Field>
      {settings.krea2SeedVarianceEnabled ? (
        <>
          <ScrubberField
            label={t('widgets.generate.krea2SeedVarianceStrength')}
            max={MAX_KREA2_SEED_VARIANCE_STRENGTH}
            min={0}
            step={0.05}
            value={settings.krea2SeedVarianceStrength}
            onChange={(value) => onCommit({ krea2SeedVarianceStrength: value })}
          />
          <ScrubberField
            formatValue={formatPercent}
            label={t('widgets.generate.krea2SeedVarianceRandomize')}
            max={100}
            min={0}
            step={1}
            value={settings.krea2SeedVarianceRandomizePercent}
            onChange={(value) => onCommit({ krea2SeedVarianceRandomizePercent: value })}
          />
        </>
      ) : null}
    </>
  );
};

const selectSeedHistory = (insights: GenerationQueueInsights) => insights.seedHistory;

/** Clicking an executed seed switches to fixed mode. */
const SeedField = ({ onCommit, settings }: RenderFieldsProps) => {
  const { t } = useTranslation();
  const seedHistory = useGenerationQueueInsights(selectSeedHistory);
  // Share expansion queries so seed counts match submission without duplicate fetches.
  const expansion = useDynamicPrompts(getEffectivePrompts(settings).positivePrompt, getDynamicPromptsConfig(settings));

  return (
    <SharedSeedField
      batchCount={settings.batchCount}
      label={t('common.seed')}
      promptCount={expansion.prompts.length}
      seed={settings.seed}
      seedBehaviour={settings.dynamicPromptsSeedBehaviour}
      seedMode={settings.seedMode}
      onCommit={onCommit}
    >
      {seedHistory.length > 0 ? (
        <HStack gap="1" pt="0.5">
          <Text color="fg.subtle" fontSize="xs">
            {t('widgets.generate.recentSeeds')}
          </Text>
          {seedHistory.map((item) => (
            <Tooltip key={item.seed} content={t('widgets.generate.useSeed', { seed: item.seed })}>
              <Box
                aria-label={t('widgets.generate.useSeed', { seed: item.seed })}
                as="button"
                bg="bg.emphasized"
                borderColor={
                  settings.seedMode !== 'random' && settings.seed === item.seed ? 'accent.solid' : 'border.subtle'
                }
                borderWidth="1px"
                boxSize="5"
                overflow="hidden"
                rounded="3px"
                onClick={() => onCommit({ seed: item.seed, seedMode: 'fixed' })}
              >
                {item.thumbnailUrl ? (
                  <Image alt="" boxSize="full" draggable={false} objectFit="cover" src={item.thumbnailUrl} />
                ) : null}
              </Box>
            </Tooltip>
          ))}
        </HStack>
      ) : null}
    </SharedSeedField>
  );
};

export const GenerateRenderSection = ({
  draft,
  onCommit,
  onCommitImmediate,
  selectedModel,
}: GenerateRenderSectionProps) => {
  const { t } = useTranslation();
  const { CanvasDenoisingStrength: CanvasDenoisingStrengthSlot, project } = useGenerationUi();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectRenderSettings);
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const uiPolicy = getGenerationUiPolicy(selectedModel);
  const familyBase = selectedModel && selectedModel.type !== 'external_image_generator' ? selectedModel.base : null;

  // Cap both input and track at the architecture ceiling without crossing the floor.
  const guidanceInputMax = uiPolicy.guidanceMax ?? GUIDANCE_INPUT_MAX;
  const guidanceSliderMax = Math.max(uiPolicy.guidanceMin, Math.min(GUIDANCE_SLIDER_MAX, guidanceInputMax));
  // Validate recalled/persisted values inline because they bypass model-selection clamps.
  const guidanceError = selectedModel ? getGuidanceBoundReason(selectedModel, settings.cfgScale) : null;

  const commitNumber = (key: 'cfgScale' | 'steps', value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: value });
  };

  const seedBadge = uiPolicy.seedVisible ? (
    <Badge>
      {settings.seedMode === 'random'
        ? t('common.seedMode.random')
        : settings.seedMode === 'fixed'
          ? settings.seed
          : t('widgets.generate.seedSummary', {
              mode: t(`common.seedMode.${settings.seedMode}`),
              seed: settings.seed,
            })}
    </Badge>
  ) : null;

  const renderFields = (strengthField: ReactNode) => (
    <>
      <GenerateFieldContextMenu
        copyValue={() => String(settings.steps)}
        isAtDefault={modelDefaults !== null && settings.steps === modelDefaults.steps}
        onReset={modelDefaults ? () => onCommit({ steps: modelDefaults.steps }) : undefined}
      >
        <ScrubberField
          defaultValue={modelDefaults?.steps}
          hint="steps"
          inputMax={Number.MAX_SAFE_INTEGER}
          label={t('widgets.generate.steps')}
          marks={modelDefaults ? [modelDefaults.steps] : undefined}
          max={STEPS_SLIDER_MAX}
          min={1}
          step={1}
          value={settings.steps}
          onChange={(steps) => commitNumber('steps', steps)}
        />
      </GenerateFieldContextMenu>
      <GenerateFieldContextMenu
        copyValue={() => String(settings.cfgScale)}
        isAtDefault={modelDefaults !== null && settings.cfgScale === modelDefaults.cfgScale}
        onReset={modelDefaults ? () => onCommit({ cfgScale: modelDefaults.cfgScale }) : undefined}
      >
        <ScrubberField
          defaultValue={modelDefaults?.cfgScale}
          error={guidanceError}
          hint="guidance"
          inputMax={guidanceInputMax}
          label={uiPolicy.guidanceLabel}
          marks={modelDefaults ? [modelDefaults.cfgScale] : undefined}
          max={guidanceSliderMax}
          min={uiPolicy.guidanceMin}
          step={0.5}
          value={settings.cfgScale}
          onChange={(cfgScale) => commitNumber('cfgScale', cfgScale)}
        />
      </GenerateFieldContextMenu>
      {strengthField}
      {familyBase === 'krea-2' ? (
        <>
          <Separator borderColor="border.subtle" />
          <GenerateConditioningRebalanceField
            settings={settings}
            onCommit={onCommit}
            onCommitImmediate={onCommitImmediate}
          />
        </>
      ) : null}
      {familyBase === 'wan' ? <WanLowNoiseGuidanceField settings={settings} onCommit={onCommit} /> : null}
      {uiPolicy.schedulerVisible ? (
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
                options={getSchedulerOptions(selectedModel, settings.scheduler)}
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
      <Separator borderColor="border.subtle" />
      {uiPolicy.seedVisible ? <SeedField settings={settings} onCommit={onCommit} /> : null}
      {familyBase === 'krea-2' ? (
        <>
          <Separator borderColor="border.subtle" />
          <Krea2SeedVarianceFields settings={settings} onCommit={onCommit} />
        </>
      ) : null}
    </>
  );

  const renderSection = (canvas: CanvasDenoisingStrength | null) => (
    <GenerateCollapsibleSection
      label={t('widgets.generate.render')}
      defaultOpen={false}
      badges={
        <>
          <Badge>
            {settings.steps} · {uiPolicy.guidanceLabel} {settings.cfgScale}
          </Badge>
          {canvas?.badges}
          {seedBadge}
        </>
      }
      sectionId="render"
    >
      <Stack gap="2" p="2">
        {renderFields(canvas?.field ?? null)}
      </Stack>
    </GenerateCollapsibleSection>
  );

  // Switching source remounts the section's subtree, as the Size section does.
  return project.invocationSourceId === 'canvas' ? (
    <CanvasDenoisingStrengthSlot>{renderSection}</CanvasDenoisingStrengthSlot>
  ) : (
    renderSection(null)
  );
};
