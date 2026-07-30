/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings, Ideogram4SamplerPreset } from '@features/generation/core/types';

import { Badge, createListCollection, Input, Stack, Switch } from '@chakra-ui/react';
import {
  IDEOGRAM4_SAMPLER_PRESETS,
  isValidKrea2RebalanceWeights,
  KREA2_REBALANCE_WEIGHT_COUNT,
  MAX_KREA2_SEED_VARIANCE_STRENGTH,
} from '@features/generation/core/settings';
// Imported by subpath rather than from the `@platform/ui` barrel: the barrel is at its
// direct-importer budget (a dev-invalidation limit), and depending on the two components
// actually used means this module only rebuilds when those change.
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { useId } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { SliderNumberField } from './shared/SliderNumberField';

interface GenerateModelFamilyFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
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
 * A switch that owns its hidden-input id. Sibling switches inside one Field.Root would
 * otherwise share an id and a label click would toggle the wrong control.
 */
const FamilySwitch = ({
  checked,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => {
  const id = useId();

  return (
    <Switch.Root
      checked={checked}
      ids={{ hiddenInput: id, label: `${id}-label` }}
      size="sm"
      onCheckedChange={(event) => onCheckedChange(event.checked)}
    >
      <Switch.HiddenInput />
      <Switch.Control _checked={{ bg: 'accent.solid' }}>
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="xs">{label}</Switch.Label>
    </Switch.Root>
  );
};

const Ideogram4Fields = ({ onCommit, settings }: Omit<GenerateModelFamilyFieldsProps, 'selectedModel'>) => {
  const { t } = useTranslation();

  return (
    <Stack gap="2" p="2">
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
        <FamilySwitch
          checked={settings.ideogram4Steps !== null}
          label={t('widgets.generate.override')}
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
        <FamilySwitch
          checked={settings.ideogram4GuidanceScale !== null}
          label={t('widgets.generate.override')}
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
        <FamilySwitch
          checked={settings.ideogram4Mu !== null}
          label={t('widgets.generate.override')}
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
    </Stack>
  );
};

const Krea2Fields = ({ onCommit, settings }: Omit<GenerateModelFamilyFieldsProps, 'selectedModel'>) => {
  const { t } = useTranslation();
  const weightsValid = isValidKrea2RebalanceWeights(settings.krea2RebalanceWeights);

  return (
    <Stack gap="2" p="2">
      <Field label={t('widgets.generate.krea2Rebalance')} helpText={t('widgets.generate.krea2RebalanceHelp')}>
        <FamilySwitch
          checked={settings.krea2RebalanceEnabled}
          label={t('widgets.generate.enabled')}
          onCheckedChange={(checked) => onCommit({ krea2RebalanceEnabled: checked })}
        />
      </Field>
      {settings.krea2RebalanceEnabled ? (
        <>
          <Field label={t('widgets.generate.krea2RebalanceMultiplier')}>
            <SliderNumberField
              ariaLabel={t('widgets.generate.krea2RebalanceMultiplier')}
              max={10}
              min={0}
              step={0.1}
              value={settings.krea2RebalanceMultiplier}
              onChange={(value) => onCommit({ krea2RebalanceMultiplier: value })}
            />
          </Field>
          <Field
            label={t('widgets.generate.krea2RebalanceWeights')}
            helpText={t('widgets.generate.krea2RebalanceWeightsHelp', { count: KREA2_REBALANCE_WEIGHT_COUNT })}
            error={
              weightsValid
                ? undefined
                : t('widgets.generate.krea2RebalanceWeightsInvalid', { count: KREA2_REBALANCE_WEIGHT_COUNT })
            }
          >
            <Input
              size="xs"
              value={settings.krea2RebalanceWeights}
              onChange={(event) => onCommit({ krea2RebalanceWeights: event.target.value })}
            />
          </Field>
        </>
      ) : null}
      <Field label={t('widgets.generate.krea2SeedVariance')} helpText={t('widgets.generate.krea2SeedVarianceHelp')}>
        <FamilySwitch
          checked={settings.krea2SeedVarianceEnabled}
          label={t('widgets.generate.enabled')}
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
    </Stack>
  );
};

const WanFields = ({ onCommit, settings }: Omit<GenerateModelFamilyFieldsProps, 'selectedModel'>) => {
  const { t } = useTranslation();

  return (
    <Stack gap="2" p="2">
      {/*
        A14B runs two experts across the schedule. Null reuses the main guidance for the
        low-noise half, which is the backend's own default.
      */}
      <Field label={t('widgets.generate.wanGuidanceLowNoise')} helpText={t('widgets.generate.wanGuidanceLowNoiseHelp')}>
        <FamilySwitch
          checked={settings.wanGuidanceScaleLowNoise !== null}
          label={t('widgets.generate.override')}
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
    </Stack>
  );
};

/**
 * Parameters that exist for exactly one model family. Keeping them here rather than in
 * GenerateAdvancedFields keeps that component about settings every base shares.
 */
export const GenerateModelFamilyFields = ({ onCommit, selectedModel, settings }: GenerateModelFamilyFieldsProps) => {
  const { t } = useTranslation();

  if (!selectedModel || selectedModel.type === 'external_image_generator') {
    return null;
  }

  const base = selectedModel.base;

  if (base !== 'ideogram-4' && base !== 'krea-2' && base !== 'wan') {
    return null;
  }

  const activeBadges = (
    <>
      {base === 'krea-2' && settings.krea2RebalanceEnabled && (
        <Badge size="xs">{t('widgets.generate.krea2Rebalance')}</Badge>
      )}
      {base === 'krea-2' && settings.krea2SeedVarianceEnabled && (
        <Badge size="xs">{t('widgets.generate.krea2SeedVariance')}</Badge>
      )}
      {base === 'ideogram-4' && <Badge size="xs">{IDEOGRAM4_PRESET_LABELS[settings.ideogram4SamplerPreset]}</Badge>}
    </>
  );

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.modelFamilySettings')}
      defaultOpen={false}
      badges={activeBadges}
      sectionId="model-family"
    >
      {base === 'ideogram-4' ? <Ideogram4Fields settings={settings} onCommit={onCommit} /> : null}
      {base === 'krea-2' ? <Krea2Fields settings={settings} onCommit={onCommit} /> : null}
      {base === 'wan' ? <WanFields settings={settings} onCommit={onCommit} /> : null}
    </GenerateCollapsibleSection>
  );
};
