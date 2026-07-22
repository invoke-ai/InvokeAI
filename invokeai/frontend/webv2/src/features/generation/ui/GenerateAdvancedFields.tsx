/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, createListCollection, HStack, Stack, Switch } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationUiPolicy } from '@features/generation/core/baseGenerationPolicies';
import { isVaeModelConfig } from '@features/generation/core/settings';
import { Field, Select } from '@platform/ui';
import { useId } from 'react';
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
  const clipSkipMax = policy.clipSkipMax ?? 0;

  const updateNumber = (key: 'cfgRescaleMultiplier' | 'clipSkip', max: number) => (value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: Math.min(max, Math.max(0, value)) });
  };

  const customVae = policy.sdVaeVisible && Boolean(settings.vae?.key);

  if (
    !policy.sdVaeVisible &&
    !policy.vaePrecisionVisible &&
    !policy.colorCompensationVisible &&
    !policy.seamlessVisible &&
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
            <Field flex="1" label={t('widgets.generate.vaePrecision')}>
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
            <Field label={t('widgets.generate.clipSkip')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.clipSkip')}
                defaultValue={modelDefaults?.clipSkip}
                max={clipSkipMax}
                min={0}
                resetLabel={t('widgets.generate.useModelDefault')}
                step={1}
                value={settings.clipSkip}
                onChange={updateNumber('clipSkip', clipSkipMax)}
              />
            </Field>
          ) : null}
          {policy.cfgRescaleVisible ? (
            <Field label={t('widgets.generate.cfgRescale')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.cfgRescale')}
                defaultValue={modelDefaults?.cfgRescaleMultiplier}
                max={0.99}
                min={0}
                resetLabel={t('widgets.generate.useModelDefaultCfgRescale')}
                step={0.05}
                value={settings.cfgRescaleMultiplier}
                onChange={updateNumber('cfgRescaleMultiplier', 0.99)}
              />
            </Field>
          ) : null}
        </Stack>
      ) : null}

      {policy.colorCompensationVisible ? (
        <Field label={t('widgets.generate.colorCompensation')} p="2">
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

      {policy.seamlessVisible ? (
        <Field label={t('widgets.generate.seamlessTiling')} p="2">
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
