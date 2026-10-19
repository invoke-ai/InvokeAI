import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { ComponentModelConfig, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Stack } from '@chakra-ui/react';
import {
  getComponentSectionPolicy,
  type ComponentPolicyContext,
  type ComponentSlotPolicy,
} from '@features/generation/core/baseGenerationPolicies';
import {
  DEFAULT_ANIMA_INPAINT_ADAPTER_WEIGHT,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  MAX_ANIMA_INPAINT_ADAPTER_WEIGHT,
  MIN_ANIMA_INPAINT_ADAPTER_WEIGHT,
} from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Field } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import { GenerationModelSelect as ModelSelect } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateComponentsSectionProps {
  draft: GenerateDraft;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const toComponentModel = (model: ModelConfig | null): ComponentModelConfig | null =>
  isModelIdentifierConfig(model) ? model : null;

const selectComponentSettings = pickGenerateSettings([
  'animaInpaintAdapterModel',
  'animaInpaintAdapterWeight',
  'clipEmbedModel',
  'clipGEmbedModel',
  'clipLEmbedModel',
  'componentSourceModel',
  'gemma2EncoderModel',
  'ideogram4UnconditionalModel',
  'mistralEncoderModel',
  'pidDecoderModel',
  'pidMode',
  'qwen35EncoderModel',
  'qwen3EncoderModel',
  'qwen3VLEncoderModel',
  'qwenVLEncoderModel',
  't5EncoderModel',
  'vae',
  'wanLowNoiseModel',
  'wanT5EncoderModel',
]);

type ComponentSettings = ReturnType<typeof selectComponentSettings>;

const getComponentPolicyContext = (
  model: GenerateModelConfig,
  settings: ComponentSettings
): ComponentPolicyContext => ({
  model,
  settings,
  selectedComponents: {
    clipEmbedModel: settings.clipEmbedModel,
    clipGEmbedModel: settings.clipGEmbedModel,
    clipLEmbedModel: settings.clipLEmbedModel,
    componentSourceModel: settings.componentSourceModel,
    mistralEncoderModel: settings.mistralEncoderModel,
    qwen3EncoderModel: settings.qwen3EncoderModel,
    qwenVLEncoderModel: settings.qwenVLEncoderModel,
    qwen3VLEncoderModel: settings.qwen3VLEncoderModel,
    qwen35EncoderModel: settings.qwen35EncoderModel,
    animaInpaintAdapterModel: settings.animaInpaintAdapterModel,
    wanT5EncoderModel: settings.wanT5EncoderModel,
    wanLowNoiseModel: settings.wanLowNoiseModel,
    ideogram4UnconditionalModel: settings.ideogram4UnconditionalModel,
    t5EncoderModel: settings.t5EncoderModel,
    pidDecoderModel: settings.pidDecoderModel,
    gemma2EncoderModel: settings.gemma2EncoderModel,
    vae: settings.vae,
  },
});

const ComponentPicker = ({
  ctx,
  onChange,
  slot,
}: {
  ctx: ComponentPolicyContext;
  onChange: (model: ModelConfig | null) => void;
  slot: ComponentSlotPolicy;
}) => {
  const value = ctx.selectedComponents[slot.key];
  const filter = useCallback((model: ModelConfig) => slot.filter?.(model, ctx) ?? false, [ctx, slot]);
  const modelTypes = useMemo(() => [...slot.modelTypes], [slot.modelTypes]);
  const selectedValue = value && (!filter || filter(value as ModelConfig)) ? value.key : null;

  return (
    <Field label={slot.label} helpText={slot.helpText}>
      <ModelSelect
        filter={slot.filter ? filter : undefined}
        isClearable
        modelTypes={modelTypes}
        placeholder={slot.placeholder ?? 'Model default'}
        value={selectedValue}
        onChange={onChange}
      />
    </Field>
  );
};

export const GenerateComponentsSection = ({ draft, onCommit, selectedModel }: GenerateComponentsSectionProps) => {
  const { t } = useTranslation();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectComponentSettings);
  const ctx = useMemo(
    () => (selectedModel ? getComponentPolicyContext(selectedModel, settings) : null),
    [selectedModel, settings]
  );

  const commitSlotValue = useCallback(
    (slot: ComponentSlotPolicy, model: ModelConfig | null) => {
      if (slot.valueKind === 'main') {
        onCommit({ [slot.key]: isMainModelConfig(model) ? model : null });
        return;
      }

      if (slot.valueKind === 'vae') {
        onCommit({ [slot.key]: isVaeModelConfig(model) ? model : null });
        return;
      }

      onCommit({ [slot.key]: toComponentModel(model) });
    },
    [onCommit]
  );

  if (!selectedModel || selectedModel.type === 'external_image_generator') {
    return null;
  }

  const policy = getComponentSectionPolicy(selectedModel, settings);

  if (policy.slots.length === 0) {
    return null;
  }

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.components')}
      defaultOpen={policy.defaultOpen}
      sectionId="components"
    >
      <Stack gap="2" p="2">
        {policy.slots.map((slot) => (
          <ComponentPickerRow key={slot.key} commitSlotValue={commitSlotValue} ctx={ctx!} slot={slot} />
        ))}
        {settings.animaInpaintAdapterModel && policy.slots.some((slot) => slot.key === 'animaInpaintAdapterModel') ? (
          <AnimaInpaintAdapterWeightField value={settings.animaInpaintAdapterWeight} onCommit={onCommit} />
        ) : null}
      </Stack>
    </GenerateCollapsibleSection>
  );
};

/** The legacy slider spanned 0 to 2 with these stops; typed values may use the node's full range. */
const ANIMA_INPAINT_ADAPTER_WEIGHT_MARKS = [0, 1, 2];

const AnimaInpaintAdapterWeightField = ({
  onCommit,
  value,
}: {
  onCommit: (patch: Partial<GenerateSettings>) => void;
  value: number;
}) => {
  const { t } = useTranslation();
  const onChange = useCallback(
    (weight: number) => {
      if (Number.isFinite(weight)) {
        onCommit({ animaInpaintAdapterWeight: weight });
      }
    },
    [onCommit]
  );

  return (
    <ScrubberField
      defaultValue={DEFAULT_ANIMA_INPAINT_ADAPTER_WEIGHT}
      inputMax={MAX_ANIMA_INPAINT_ADAPTER_WEIGHT}
      inputMin={MIN_ANIMA_INPAINT_ADAPTER_WEIGHT}
      label={t('widgets.generate.animaInpaintAdapterWeight')}
      marks={ANIMA_INPAINT_ADAPTER_WEIGHT_MARKS}
      max={2}
      min={0}
      step={0.05}
      value={value}
      onChange={onChange}
    />
  );
};

const ComponentPickerRow = ({
  commitSlotValue,
  ctx,
  slot,
}: {
  commitSlotValue: (slot: ComponentSlotPolicy, model: ModelConfig | null) => void;
  ctx: ComponentPolicyContext;
  slot: ComponentSlotPolicy;
}) => {
  const onChange = useCallback((model: ModelConfig | null) => commitSlotValue(slot, model), [commitSlotValue, slot]);

  return <ComponentPicker ctx={ctx} slot={slot} onChange={onChange} />;
};
