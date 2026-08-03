/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, createListCollection, HStack, Stack, Switch } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationUiPolicy } from '@features/generation/core/baseGenerationPolicies';
import { isPidMode, MAX_PID_STEPS, MIN_PID_STEPS } from '@features/generation/core/pid';
import {
  DEFAULT_HIDIFFUSION_T1_RATIO,
  DEFAULT_HIDIFFUSION_T2_RATIO,
  isVaeModelConfig,
  MAX_HIDIFFUSION_RATIO,
  MIN_HIDIFFUSION_T1_RATIO,
} from '@features/generation/core/settings';
import { Field, Select } from '@platform/ui';
import { useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerationModelSelect as ModelSelect } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';
import { SliderNumberField } from './shared/SliderNumberField';

interface GenerateAdvancedFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const VAE_PRECISION_COLLECTION = createListCollection({
  items: [
    { label: 'FP16', value: 'fp16' },
    { label: 'FP32', value: 'fp32' },
  ] as const,
});

const SeamlessSwitch = ({
  checked,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => {
  // Both axis switches share one Field.Root, which would hand them the same
  // hidden-input id — explicit ids keep each label bound to its own input.
  const id = useId();

  return (
    <Switch.Root
      checked={checked}
      flex="1"
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

const AdvancedSwitch = ({
  checked,
  disabled = false,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => (
  <Switch.Root
    checked={checked}
    disabled={disabled}
    size="sm"
    onCheckedChange={(event) => onCheckedChange(event.checked)}
  >
    <Switch.HiddenInput />
    <Switch.Control _checked={{ bg: 'accent.solid' }}>
      <Switch.Thumb />
    </Switch.Control>
    <Switch.Label srOnly>{label}</Switch.Label>
  </Switch.Root>
);

export const GenerateAdvancedFields = ({
  onCommit,
  onCommitImmediate,
  selectedModel,
  settings,
}: GenerateAdvancedFieldsProps) => {
  const { t } = useTranslation();
  const modelBase = selectedModel?.base;
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationUiPolicy(selectedModel, settings);
  const pidHelpText =
    settings.pidMode === 'fit'
      ? t('widgets.generate.pidFitHelp')
      : settings.pidMode === 'native'
        ? t('widgets.generate.pidNativeHelp')
        : t('widgets.generate.pidHelp');
  const pidModeCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.generate.pidOff'), value: 'off' },
          { label: t('widgets.generate.pidFit'), value: 'fit' },
          { label: t('widgets.generate.pidNative'), value: 'native' },
        ],
      }),
    [t]
  );
  const clipSkipMax = policy.clipSkipMax ?? 0;

  const updateNumber =
    (
      key: 'cfgRescaleMultiplier' | 'clipSkip' | 'hiDiffusionT1Ratio' | 'hiDiffusionT2Ratio',
      min: number,
      max: number
    ) =>
    (value: number) => {
      if (!Number.isFinite(value)) {
        return;
      }

      onCommit({ [key]: Math.min(max, Math.max(min, value)) });
    };

  const customVae = policy.sdVaeVisible && Boolean(settings.vae?.key);

  if (
    !policy.sdVaeVisible &&
    !policy.vaePrecisionVisible &&
    !policy.colorCompensationVisible &&
    !policy.seamlessVisible &&
    !policy.hiDiffusionVisible &&
    !policy.clipSkipMax &&
    !policy.cfgRescaleVisible
  ) {
    return null;
  }

  const badges = (
    <>
      {settings.seamlessXAxis && <Badge size="xs">{t('widgets.generate.tileX')}</Badge>}
      {settings.seamlessYAxis && <Badge size="xs">{t('widgets.generate.tileY')}</Badge>}
      {settings.colorCompensation && <Badge size="xs">{t('widgets.generate.colorCompensation')}</Badge>}
      {settings.hiDiffusionEnabled && <Badge size="xs">{t('widgets.generate.hiDiffusion')}</Badge>}
      {customVae && (
        <Badge maxW="32" size="xs" truncate>
          {settings.vae?.name}
        </Badge>
      )}
    </>
  );

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.advanced')}
      defaultOpen={false}
      badges={badges}
      sectionId="advanced"
    >
      {policy.sdVaeVisible || policy.vaePrecisionVisible ? (
        <HStack alignItems="flex-start" gap="2" p="2">
          {policy.sdVaeVisible ? (
            <Field
              flex="2"
              hint="vae"
              label={t('widgets.generate.vae')}
              helpText={settings.vae ? undefined : t('widgets.generate.usingBundledVae')}
            >
              <HStack gap="1">
                <ModelSelect
                  filter={(model) => model.base === modelBase}
                  modelTypes={['vae']}
                  size="xs"
                  placeholder={t('widgets.generate.modelDefault')}
                  value={settings.vae?.key ?? null}
                  onChange={(model) => onCommitImmediate({ vae: isVaeModelConfig(model) ? model : null })}
                />
                {settings.vae ? (
                  <ModelDefaultButton
                    label={t('widgets.generate.useModelDefaultVae')}
                    onClick={() => onCommitImmediate({ vae: null })}
                  />
                ) : null}
              </HStack>
            </Field>
          ) : null}
          {policy.vaePrecisionVisible ? (
            <Field flex="1" hint="vaePrecision" label={t('widgets.generate.vaePrecision')}>
              <HStack gap="1">
                <Select
                  aria-label={t('widgets.generate.vaePrecision')}
                  collection={VAE_PRECISION_COLLECTION}
                  flex="1"
                  size="xs"
                  value={[settings.vaePrecision]}
                  onValueChange={({ value }) => {
                    const vaePrecision = value[0];

                    if (vaePrecision === 'fp16' || vaePrecision === 'fp32') {
                      onCommit({ vaePrecision });
                    }
                  }}
                />
                {modelDefaults && settings.vaePrecision !== modelDefaults.vaePrecision ? (
                  <ModelDefaultButton
                    label={t('widgets.generate.useModelDefaultVaePrecision')}
                    onClick={() => onCommit({ vaePrecision: modelDefaults.vaePrecision })}
                  />
                ) : null}
              </HStack>
            </Field>
          ) : null}
        </HStack>
      ) : null}

      {policy.clipSkipMax || policy.cfgRescaleVisible ? (
        <Stack gap="2" p="2">
          {policy.clipSkipMax ? (
            <Field hint="clipSkip" label={t('widgets.generate.clipSkip')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.clipSkip')}
                defaultValue={modelDefaults?.clipSkip}
                max={clipSkipMax}
                min={0}
                resetLabel={t('widgets.generate.useModelDefault')}
                step={1}
                value={settings.clipSkip}
                onChange={updateNumber('clipSkip', 0, clipSkipMax)}
              />
            </Field>
          ) : null}
          {policy.cfgRescaleVisible ? (
            <Field hint="cfgRescale" label={t('widgets.generate.cfgRescale')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.cfgRescale')}
                defaultValue={modelDefaults?.cfgRescaleMultiplier}
                max={0.99}
                min={0}
                resetLabel={t('widgets.generate.useModelDefaultCfgRescale')}
                step={0.05}
                value={settings.cfgRescaleMultiplier}
                onChange={updateNumber('cfgRescaleMultiplier', 0, 0.99)}
              />
            </Field>
          ) : null}
        </Stack>
      ) : null}

      {policy.hiDiffusionVisible ? (
        <Stack gap="2" p="2">
          <Field hint="hidiffusion" label={t('widgets.generate.hiDiffusion')}>
            <AdvancedSwitch
              checked={settings.hiDiffusionEnabled}
              label={t('widgets.generate.hiDiffusion')}
              onCheckedChange={(checked) => onCommit({ hiDiffusionEnabled: checked })}
            />
          </Field>
          <HStack alignItems="flex-start" gap="2">
            <Field flex="1" hint="hidiffusionRauNet" label={t('widgets.generate.hiDiffusionRauNet')}>
              <AdvancedSwitch
                checked={settings.hiDiffusionRauNetEnabled}
                disabled={!settings.hiDiffusionEnabled}
                label={t('widgets.generate.hiDiffusionRauNet')}
                onCheckedChange={(checked) => onCommit({ hiDiffusionRauNetEnabled: checked })}
              />
            </Field>
            <Field flex="1" hint="hidiffusionWindowAttn" label={t('widgets.generate.hiDiffusionWindowAttn')}>
              <AdvancedSwitch
                checked={settings.hiDiffusionWindowAttentionEnabled}
                disabled={!settings.hiDiffusionEnabled}
                label={t('widgets.generate.hiDiffusionWindowAttn')}
                onCheckedChange={(checked) => onCommit({ hiDiffusionWindowAttentionEnabled: checked })}
              />
            </Field>
          </HStack>
          <Field hint="hidiffusionT1Ratio" label={t('widgets.generate.hiDiffusionT1Ratio')}>
            <SliderNumberField
              ariaLabel={t('widgets.generate.hiDiffusionT1Ratio')}
              defaultValue={DEFAULT_HIDIFFUSION_T1_RATIO}
              disabled={!settings.hiDiffusionEnabled}
              max={MAX_HIDIFFUSION_RATIO}
              min={MIN_HIDIFFUSION_T1_RATIO}
              resetLabel={t('widgets.generate.useModelDefault')}
              step={0.05}
              value={settings.hiDiffusionT1Ratio}
              onChange={updateNumber('hiDiffusionT1Ratio', MIN_HIDIFFUSION_T1_RATIO, MAX_HIDIFFUSION_RATIO)}
            />
          </Field>
          <Field hint="hidiffusionT2Ratio" label={t('widgets.generate.hiDiffusionT2Ratio')}>
            <SliderNumberField
              ariaLabel={t('widgets.generate.hiDiffusionT2Ratio')}
              defaultValue={DEFAULT_HIDIFFUSION_T2_RATIO}
              disabled={!settings.hiDiffusionEnabled}
              max={MAX_HIDIFFUSION_RATIO}
              min={0}
              resetLabel={t('widgets.generate.useModelDefault')}
              step={0.05}
              value={settings.hiDiffusionT2Ratio}
              onChange={updateNumber('hiDiffusionT2Ratio', 0, MAX_HIDIFFUSION_RATIO)}
            />
          </Field>
        </Stack>
      ) : null}

      {policy.colorCompensationVisible ? (
        <Field hint="colorCompensation" label={t('widgets.generate.colorCompensation')} p="2">
          <Switch.Root
            checked={settings.colorCompensation}
            size="sm"
            onCheckedChange={(event) => onCommit({ colorCompensation: event.checked })}
          >
            <Switch.HiddenInput />
            <Switch.Control _checked={{ bg: 'accent.solid' }}>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Root>
        </Field>
      ) : null}

      {policy.pidVisible ? (
        <HStack alignItems="flex-start" gap="2" p="2">
          <Field flex="2" hint="pidMode" label={t('widgets.generate.pid')} helpText={pidHelpText}>
            <Select
              aria-label={t('widgets.generate.pid')}
              collection={pidModeCollection}
              size="xs"
              value={[settings.pidMode]}
              onValueChange={({ value }) => {
                const mode = value[0];

                if (isPidMode(mode)) {
                  // Committed immediately: switching to or from native changes the dimension
                  // grid, and a debounced write would let a now-invalid size linger.
                  onCommitImmediate({ pidMode: mode });
                }
              }}
            />
          </Field>
          {settings.pidMode === 'off' ? null : (
            <Field flex="1" label={t('widgets.generate.pidSteps')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.pidSteps')}
                min={MIN_PID_STEPS}
                max={MAX_PID_STEPS}
                step={1}
                value={settings.pidSteps}
                onChange={(value) => onCommit({ pidSteps: value })}
              />
            </Field>
          )}
        </HStack>
      ) : null}

      {policy.seamlessVisible ? (
        <Field hint="seamlessTiling" label={t('widgets.generate.seamlessTiling')} p="2">
          <HStack gap="4">
            <SeamlessSwitch
              checked={settings.seamlessXAxis}
              label={t('widgets.generate.xAxis')}
              onCheckedChange={(checked) => onCommit({ seamlessXAxis: checked })}
            />
            <SeamlessSwitch
              checked={settings.seamlessYAxis}
              label={t('widgets.generate.yAxis')}
              onCheckedChange={(checked) => onCommit({ seamlessYAxis: checked })}
            />
          </HStack>
        </Field>
      ) : null}
    </GenerateCollapsibleSection>
  );
};
