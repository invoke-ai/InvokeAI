/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type {
  GenerateModelConfig,
  GenerateSettings,
  QwenImageEncoderQuantization,
} from '@features/generation/core/types';

import { chakra, createListCollection, Stack, useRecipe } from '@chakra-ui/react';
import {
  getDefaultGenerateSettings,
  getGenerationUiPolicy,
  getKrea2EditLoraState,
} from '@features/generation/core/baseGenerationPolicies';
import { isPidMode, MAX_PID_STEPS, MIN_PID_STEPS } from '@features/generation/core/pid';
import {
  isQwenImageEncoderQuantization,
  isVaeModelConfig,
  MAX_HIDIFFUSION_RATIO,
  QWEN_IMAGE_ENCODER_QUANTIZATIONS,
} from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Field, Select } from '@platform/ui';
import { FEATURE_HINT_FIELD_PROPS, FeatureHint, type FeatureHintId } from '@platform/ui/hints';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { fieldLabelRecipe } from '@theme/recipes';
import { useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import { GenerationModelSelect as ModelSelect } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';
import { GenerateIssueNotes } from './shared/GenerateIssues';
import { GenerateToggleSwitch } from './shared/GenerateToggleSwitch';

interface GenerateAdvancedFieldsProps {
  draft: GenerateDraft;
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

const ENCODER_QUANTIZATION_LABEL_KEYS: Record<QwenImageEncoderQuantization, string> = {
  int8: 'widgets.generate.encoderQuantizationInt8',
  nf4: 'widgets.generate.encoderQuantizationNf4',
  none: 'widgets.generate.encoderQuantizationNone',
};

const selectAdvancedSettings = pickGenerateSettings([
  'cfgRescaleMultiplier',
  'clipSkip',
  'colorCompensation',
  'hiDiffusionEnabled',
  'hiDiffusionManualRatios',
  'hiDiffusionRauNetEnabled',
  'hiDiffusionT1Ratio',
  'hiDiffusionT2Ratio',
  'hiDiffusionWindowAttentionEnabled',
  'krea2ReferenceKvCache',
  'pidMode',
  'pidSteps',
  'qwenImageEncoderQuantization',
  'seamlessXAxis',
  'seamlessYAxis',
  'vae',
  'vaePrecision',
]);

/** On/off settings whose model default is their reset. */
type DefaultedSwitchKey =
  | 'colorCompensation'
  | 'hiDiffusionEnabled'
  | 'hiDiffusionRauNetEnabled'
  | 'hiDiffusionWindowAttentionEnabled'
  | 'krea2ReferenceKvCache'
  | 'seamlessXAxis'
  | 'seamlessYAxis';

const KREA2_REFERENCE_KV_CACHE_HELP_KEYS = {
  noEditReferences: 'widgets.generate.krea2ReferenceKvCacheNeedsEditReferences',
  noLora: 'widgets.generate.krea2ReferenceKvCacheNeedsLora',
  ready: 'widgets.generate.krea2ReferenceKvCacheHelp',
} as const;

export const GenerateAdvancedFields = ({
  draft,
  onCommit,
  onCommitImmediate,
  selectedModel,
}: GenerateAdvancedFieldsProps) => {
  const { t } = useTranslation();
  const fieldLabel = useRecipe({ recipe: fieldLabelRecipe });
  const seamlessLabelId = useId();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectAdvancedSettings);
  // A string, so editing a reference or a concept re-renders this section only when the K/V cache starts or stops
  // applying.
  const krea2EditLoraState = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, (current: GenerateSettings) =>
    getKrea2EditLoraState(selectedModel, current)
  );
  const krea2ReferenceKvCacheApplies = krea2EditLoraState === 'ready';
  const modelBase = selectedModel?.base;
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationUiPolicy(selectedModel);
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
  const encoderQuantizationCollection = useMemo(
    () =>
      createListCollection({
        items: QWEN_IMAGE_ENCODER_QUANTIZATIONS.map((value) => ({
          label: t(ENCODER_QUANTIZATION_LABEL_KEYS[value]),
          value,
        })),
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

  const manualHiDiffusionRatios = {
    t1: settings.hiDiffusionT1Ratio ?? settings.hiDiffusionManualRatios?.t1 ?? 0.4,
    t2: settings.hiDiffusionT2Ratio ?? settings.hiDiffusionManualRatios?.t2 ?? 0,
  };
  const defaultHiDiffusionT1Ratio = Math.min(MAX_HIDIFFUSION_RATIO, Math.max(0.4, manualHiDiffusionRatios.t2));
  const setHiDiffusionRatiosAutomatic = (automatic: boolean) => {
    onCommit(
      automatic
        ? {
            hiDiffusionManualRatios: manualHiDiffusionRatios,
            hiDiffusionT1Ratio: null,
            hiDiffusionT2Ratio: null,
          }
        : {
            hiDiffusionT1Ratio: manualHiDiffusionRatios.t1,
            hiDiffusionT2Ratio: manualHiDiffusionRatios.t2,
          }
    );
  };

  const hiDiffusionRatiosAutomatic = settings.hiDiffusionT1Ratio === null && settings.hiDiffusionT2Ratio === null;
  const hiDiffusionRatiosDisabled = !settings.hiDiffusionEnabled || !settings.hiDiffusionRauNetEnabled;
  const hiDiffusionT1Disabled = hiDiffusionRatiosDisabled || settings.hiDiffusionT1Ratio === null;
  const hiDiffusionT2Disabled = hiDiffusionRatiosDisabled || settings.hiDiffusionT2Ratio === null;
  const onOff = (value: boolean) => t(value ? 'commandPalette.settings.on' : 'commandPalette.settings.off');
  /** An on/off setting with the model's default value: a changed one shows the dot that puts it back. */
  const defaultedSwitch = ({
    key,
    checked = settings[key],
    disabled = false,
    helpText,
    hint,
    label,
  }: {
    checked?: boolean;
    disabled?: boolean;
    helpText?: string;
    hint?: FeatureHintId;
    key: DefaultedSwitchKey;
    label: string;
  }) => (
    <GenerateFieldContextMenu
      fieldLabel={label}
      copyValue={() => onOff(checked)}
      // A disabled switch is not sent, so it is no change to undo.
      isAtDefault={disabled || modelDefaults === null || checked === modelDefaults[key]}
      placement="switch"
      onReset={modelDefaults ? () => onCommit({ [key]: modelDefaults[key] }) : undefined}
    >
      <GenerateToggleSwitch
        checked={checked}
        disabled={disabled}
        helpText={helpText}
        hint={hint}
        label={label}
        onCheckedChange={(next) => onCommit({ [key]: next })}
      />
    </GenerateFieldContextMenu>
  );

  const customVae = policy.sdVaeVisible && Boolean(settings.vae?.key);

  if (
    !policy.sdVaeVisible &&
    !policy.vaePrecisionVisible &&
    !policy.colorCompensationVisible &&
    !policy.seamlessVisible &&
    !policy.hiDiffusionVisible &&
    !policy.clipSkipMax &&
    !policy.cfgRescaleVisible &&
    !policy.pidVisible &&
    !policy.encoderQuantizationVisible &&
    !policy.krea2ReferenceKvCacheVisible
  ) {
    return null;
  }

  // Count user decisions, not modified settings keys, and only those the section shows (and so can reset).
  const overrideCount = [
    customVae,
    policy.vaePrecisionVisible && modelDefaults !== null && settings.vaePrecision !== modelDefaults.vaePrecision,
    clipSkipMax > 0 && settings.clipSkip !== (modelDefaults?.clipSkip ?? 0),
    policy.cfgRescaleVisible && settings.cfgRescaleMultiplier !== (modelDefaults?.cfgRescaleMultiplier ?? 0),
    policy.seamlessVisible && (settings.seamlessXAxis || settings.seamlessYAxis),
    policy.colorCompensationVisible && settings.colorCompensation,
    policy.hiDiffusionVisible && settings.hiDiffusionEnabled,
    policy.pidVisible && settings.pidMode !== 'off',
    policy.encoderQuantizationVisible && settings.qwenImageEncoderQuantization !== 'none',
    // The graph sends the switch only with edit references and a LoRA, so it is no override without them.
    policy.krea2ReferenceKvCacheVisible && krea2ReferenceKvCacheApplies && settings.krea2ReferenceKvCache,
  ].filter(Boolean).length;

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.advanced')}
      defaultOpen={false}
      isChanged={overrideCount > 0}
      sectionId="advanced"
      summary={overrideCount > 0 ? [t('widgets.generate.overridesCount', { count: overrideCount })] : undefined}
    >
      <Stack _empty={{ display: 'none' }}>
        <GenerateIssueNotes section="advanced" />
      </Stack>
      {policy.sdVaeVisible || policy.vaePrecisionVisible ? (
        // Stacked: side by side, two inside labels would leave each value too little room.
        <Stack gap="2.5">
          {policy.sdVaeVisible ? (
            <GenerateFieldContextMenu
              fieldLabel={t('widgets.generate.vae')}
              copyValue={() => settings.vae?.name ?? t('widgets.generate.modelDefault')}
              isAtDefault={!settings.vae}
              onReset={() => onCommitImmediate({ vae: null })}
            >
              <Field
                hint="vae"
                label={t('widgets.generate.vae')}
                labelPlacement="inside"
                helpText={settings.vae ? undefined : t('widgets.generate.usingBundledVae')}
              >
                <ModelSelect
                  filter={(model) => model.base === modelBase}
                  modelTypes={['vae']}
                  placeholder={t('widgets.generate.modelDefault')}
                  relatedTo={selectedModel?.key ?? null}
                  value={settings.vae?.key ?? null}
                  onChange={(model) => onCommitImmediate({ vae: isVaeModelConfig(model) ? model : null })}
                />
              </Field>
            </GenerateFieldContextMenu>
          ) : null}
          {policy.vaePrecisionVisible ? (
            <GenerateFieldContextMenu
              fieldLabel={t('widgets.generate.vaePrecision')}
              copyValue={() => settings.vaePrecision.toUpperCase()}
              isAtDefault={modelDefaults !== null && settings.vaePrecision === modelDefaults.vaePrecision}
              onReset={modelDefaults ? () => onCommit({ vaePrecision: modelDefaults.vaePrecision }) : undefined}
            >
              <Field hint="vaePrecision" label={t('widgets.generate.vaePrecision')} labelPlacement="inside">
                <Select
                  aria-label={t('widgets.generate.vaePrecision')}
                  collection={VAE_PRECISION_COLLECTION}
                  value={[settings.vaePrecision]}
                  onValueChange={({ value }) => {
                    const vaePrecision = value[0];

                    if (vaePrecision === 'fp16' || vaePrecision === 'fp32') {
                      onCommit({ vaePrecision });
                    }
                  }}
                />
              </Field>
            </GenerateFieldContextMenu>
          ) : null}
        </Stack>
      ) : null}

      {policy.clipSkipMax || policy.cfgRescaleVisible ? (
        <Stack gap="2.5">
          {policy.clipSkipMax ? (
            <GenerateFieldContextMenu
              fieldLabel={t('widgets.generate.clipSkip')}
              copyValue={() => String(settings.clipSkip)}
              isAtDefault={modelDefaults !== null && settings.clipSkip === modelDefaults.clipSkip}
              onReset={modelDefaults ? () => onCommit({ clipSkip: modelDefaults.clipSkip }) : undefined}
            >
              <ScrubberField
                defaultValue={modelDefaults?.clipSkip}
                hint="clipSkip"
                label={t('widgets.generate.clipSkip')}
                max={clipSkipMax}
                min={0}
                step={1}
                value={settings.clipSkip}
                onChange={updateNumber('clipSkip', 0, clipSkipMax)}
              />
            </GenerateFieldContextMenu>
          ) : null}
          {policy.cfgRescaleVisible ? (
            <GenerateFieldContextMenu
              fieldLabel={t('widgets.generate.cfgRescale')}
              copyValue={() => String(settings.cfgRescaleMultiplier)}
              isAtDefault={
                modelDefaults !== null && settings.cfgRescaleMultiplier === modelDefaults.cfgRescaleMultiplier
              }
              onReset={
                modelDefaults ? () => onCommit({ cfgRescaleMultiplier: modelDefaults.cfgRescaleMultiplier }) : undefined
              }
            >
              <ScrubberField
                defaultValue={modelDefaults?.cfgRescaleMultiplier}
                hint="cfgRescale"
                label={t('widgets.generate.cfgRescale')}
                max={0.99}
                min={0}
                step={0.05}
                value={settings.cfgRescaleMultiplier}
                onChange={updateNumber('cfgRescaleMultiplier', 0, 0.99)}
              />
            </GenerateFieldContextMenu>
          ) : null}
        </Stack>
      ) : null}

      {policy.hiDiffusionVisible ? (
        <Stack gap="2.5">
          {defaultedSwitch({
            hint: 'hidiffusion',
            key: 'hiDiffusionEnabled',
            label: t('widgets.generate.hiDiffusion'),
          })}
          {defaultedSwitch({
            disabled: !settings.hiDiffusionEnabled,
            hint: 'hidiffusionRauNet',
            key: 'hiDiffusionRauNetEnabled',
            label: t('widgets.generate.hiDiffusionRauNet'),
          })}
          {defaultedSwitch({
            disabled: !settings.hiDiffusionEnabled,
            hint: 'hidiffusionWindowAttn',
            key: 'hiDiffusionWindowAttentionEnabled',
            label: t('widgets.generate.hiDiffusionWindowAttn'),
          })}
          {/* Automatic is the default: no manual ratios. */}
          <GenerateFieldContextMenu
            fieldLabel={t('widgets.generate.hiDiffusionRatiosAuto')}
            copyValue={() => onOff(hiDiffusionRatiosAutomatic)}
            isAtDefault={hiDiffusionRatiosAutomatic || hiDiffusionRatiosDisabled}
            placement="switch"
            onReset={() => setHiDiffusionRatiosAutomatic(true)}
          >
            <GenerateToggleSwitch
              checked={hiDiffusionRatiosAutomatic}
              disabled={hiDiffusionRatiosDisabled}
              hint="hidiffusionRatiosAuto"
              label={t('widgets.generate.hiDiffusionRatiosAuto')}
              onCheckedChange={setHiDiffusionRatiosAutomatic}
            />
          </GenerateFieldContextMenu>
          <GenerateFieldContextMenu
            fieldLabel={t('widgets.generate.hiDiffusionT1Ratio')}
            copyValue={() => String(settings.hiDiffusionT1Ratio ?? t('widgets.generate.hiDiffusionRatioAutoValue'))}
            // A disabled ratio is not sent (HiDiffusion or RAU-Net off, or automatic ratios), so it is no change to undo.
            isAtDefault={hiDiffusionT1Disabled || manualHiDiffusionRatios.t1 === defaultHiDiffusionT1Ratio}
            onReset={() => onCommit({ hiDiffusionT1Ratio: defaultHiDiffusionT1Ratio })}
          >
            <ScrubberField
              defaultValue={defaultHiDiffusionT1Ratio}
              disabled={hiDiffusionT1Disabled}
              hint="hidiffusionT1Ratio"
              label={t('widgets.generate.hiDiffusionT1Ratio')}
              max={MAX_HIDIFFUSION_RATIO}
              min={manualHiDiffusionRatios.t2}
              step={0.05}
              value={manualHiDiffusionRatios.t1}
              onChange={updateNumber('hiDiffusionT1Ratio', manualHiDiffusionRatios.t2, MAX_HIDIFFUSION_RATIO)}
            />
          </GenerateFieldContextMenu>
          <GenerateFieldContextMenu
            fieldLabel={t('widgets.generate.hiDiffusionT2Ratio')}
            copyValue={() => String(settings.hiDiffusionT2Ratio ?? t('widgets.generate.hiDiffusionRatioAutoValue'))}
            isAtDefault={hiDiffusionT2Disabled || manualHiDiffusionRatios.t2 === 0}
            onReset={() => onCommit({ hiDiffusionT2Ratio: 0 })}
          >
            <ScrubberField
              defaultValue={0}
              disabled={hiDiffusionT2Disabled}
              hint="hidiffusionT2Ratio"
              label={t('widgets.generate.hiDiffusionT2Ratio')}
              max={manualHiDiffusionRatios.t1}
              min={0}
              step={0.05}
              value={manualHiDiffusionRatios.t2}
              onChange={updateNumber('hiDiffusionT2Ratio', 0, manualHiDiffusionRatios.t1)}
            />
          </GenerateFieldContextMenu>
        </Stack>
      ) : null}

      {policy.colorCompensationVisible
        ? defaultedSwitch({
            hint: 'colorCompensation',
            key: 'colorCompensation',
            label: t('widgets.generate.colorCompensation'),
          })
        : null}

      {policy.pidVisible ? (
        <Stack gap="2.5">
          <GenerateFieldContextMenu
            fieldLabel={t('widgets.generate.pid')}
            copyValue={() => pidModeCollection.items.find((item) => item.value === settings.pidMode)?.label ?? ''}
            isAtDefault={modelDefaults !== null && settings.pidMode === modelDefaults.pidMode}
            // Immediately, as a pick does: native mode changes the valid dimension grid.
            onReset={modelDefaults ? () => onCommitImmediate({ pidMode: modelDefaults.pidMode }) : undefined}
          >
            <Field hint="pidMode" label={t('widgets.generate.pid')} labelPlacement="inside" helpText={pidHelpText}>
              <Select
                aria-label={t('widgets.generate.pid')}
                collection={pidModeCollection}
                value={[settings.pidMode]}
                onValueChange={({ value }) => {
                  const mode = value[0];

                  if (isPidMode(mode)) {
                    // Commit immediately because native mode changes the valid dimension grid.
                    onCommitImmediate({ pidMode: mode });
                  }
                }}
              />
            </Field>
          </GenerateFieldContextMenu>
          {settings.pidMode === 'off' ? null : (
            <GenerateFieldContextMenu
              fieldLabel={t('widgets.generate.pidSteps')}
              copyValue={() => String(settings.pidSteps)}
              isAtDefault={modelDefaults !== null && settings.pidSteps === modelDefaults.pidSteps}
              onReset={modelDefaults ? () => onCommit({ pidSteps: modelDefaults.pidSteps }) : undefined}
            >
              <ScrubberField
                defaultValue={modelDefaults?.pidSteps}
                label={t('widgets.generate.pidSteps')}
                max={MAX_PID_STEPS}
                min={MIN_PID_STEPS}
                step={1}
                value={settings.pidSteps}
                onChange={(value) => onCommit({ pidSteps: value })}
              />
            </GenerateFieldContextMenu>
          )}
        </Stack>
      ) : null}

      {policy.encoderQuantizationVisible ? (
        <GenerateFieldContextMenu
          fieldLabel={t('widgets.generate.encoderQuantization')}
          copyValue={() =>
            encoderQuantizationCollection.items.find((item) => item.value === settings.qwenImageEncoderQuantization)
              ?.label ?? ''
          }
          isAtDefault={
            modelDefaults !== null &&
            settings.qwenImageEncoderQuantization === modelDefaults.qwenImageEncoderQuantization
          }
          onReset={
            modelDefaults
              ? () => onCommit({ qwenImageEncoderQuantization: modelDefaults.qwenImageEncoderQuantization })
              : undefined
          }
        >
          <Field
            hint="encoderQuantization"
            label={t('widgets.generate.encoderQuantization')}
            labelPlacement="inside"
            helpText={t('widgets.generate.encoderQuantizationHelp')}
          >
            <Select
              aria-label={t('widgets.generate.encoderQuantization')}
              collection={encoderQuantizationCollection}
              value={[settings.qwenImageEncoderQuantization]}
              onValueChange={({ value }) => {
                const quantization = value[0];

                if (isQwenImageEncoderQuantization(quantization)) {
                  onCommit({ qwenImageEncoderQuantization: quantization });
                }
              }}
            />
          </Field>
        </GenerateFieldContextMenu>
      ) : null}

      {policy.krea2ReferenceKvCacheVisible
        ? // Shown off while it is not sent; the stored value stays and is back once it applies again.
          defaultedSwitch({
            checked: krea2ReferenceKvCacheApplies && settings.krea2ReferenceKvCache,
            disabled: !krea2ReferenceKvCacheApplies,
            helpText: t(KREA2_REFERENCE_KV_CACHE_HELP_KEYS[krea2EditLoraState]),
            hint: 'krea2ReferenceKvCache',
            key: 'krea2ReferenceKvCache',
            label: t('widgets.generate.krea2ReferenceKvCache'),
          })
        : null}

      {policy.seamlessVisible ? (
        // A heading over its rows: more room below it than between the rows, so it does not read as a third row.
        <Stack aria-labelledby={seamlessLabelId} gap="3" role="group" {...FEATURE_HINT_FIELD_PROPS}>
          {/* The hint trigger takes over its child's id, so it wraps the text inside the named heading. */}
          <chakra.span css={fieldLabel()} id={seamlessLabelId}>
            <FeatureHint hint="seamlessTiling">
              <chakra.span>{t('widgets.generate.seamlessTiling')}</chakra.span>
            </FeatureHint>
          </chakra.span>
          <Stack gap="2.5">
            {defaultedSwitch({ key: 'seamlessXAxis', label: t('widgets.generate.xAxis') })}
            {defaultedSwitch({ key: 'seamlessYAxis', label: t('widgets.generate.yAxis') })}
          </Stack>
        </Stack>
      ) : null}
    </GenerateCollapsibleSection>
  );
};
