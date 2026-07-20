import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { ComponentModelConfig, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Stack } from '@chakra-ui/react';
import {
  getComponentSectionPolicy,
  type ComponentPolicyContext,
  type ComponentSlotPolicy,
} from '@features/generation/core/baseGenerationPolicies';
import { isMainModelConfig, isModelIdentifierConfig, isVaeModelConfig } from '@features/generation/core/settings';
import { Field } from '@platform/ui';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerationModelSelect as ModelSelect } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateComponentsSectionProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const toComponentModel = (model: ModelConfig | null): ComponentModelConfig | null =>
  isModelIdentifierConfig(model) ? model : null;

const getComponentPolicyContext = (model: GenerateModelConfig, settings: GenerateSettings): ComponentPolicyContext => ({
  model,
  settings,
  selectedComponents: {
    clipEmbedModel: settings.clipEmbedModel,
    clipGEmbedModel: settings.clipGEmbedModel,
    clipLEmbedModel: settings.clipLEmbedModel,
    componentSourceModel: settings.componentSourceModel,
    qwen3EncoderModel: settings.qwen3EncoderModel,
    qwenVLEncoderModel: settings.qwenVLEncoderModel,
    t5EncoderModel: settings.t5EncoderModel,
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
        size="xs"
        value={selectedValue}
        onChange={onChange}
      />
    </Field>
  );
};

export const GenerateComponentsSection = ({ onCommit, selectedModel, settings }: GenerateComponentsSectionProps) => {
  const { t } = useTranslation();
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
    <GenerateCollapsibleSection label={t('widgets.generate.components')} defaultOpen={policy.defaultOpen}>
      <Stack gap="2" p="2">
        {policy.slots.map((slot) => (
          <ComponentPickerRow key={slot.key} commitSlotValue={commitSlotValue} ctx={ctx!} slot={slot} />
        ))}
      </Stack>
    </GenerateCollapsibleSection>
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
