import type { ComponentModelConfig, GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

import { Stack } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import {
  getComponentSectionPolicy,
  type ComponentPolicyContext,
  type ComponentSlotPolicy,
} from '@workbench/generation/baseGenerationPolicies';
import { isMainModelConfig, isModelIdentifierConfig, isVaeModelConfig } from '@workbench/generation/settings';
import { ModelSelect } from '@workbench/models/components';

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
  const filter = slot.filter ? (model: ModelConfig) => slot.filter?.(model, ctx) ?? false : undefined;
  const selectedValue = value && (!filter || filter(value as ModelConfig)) ? value.key : null;

  return (
    <Field label={slot.label} helpText={slot.helpText}>
      <ModelSelect
        filter={filter}
        modelTypes={[...slot.modelTypes]}
        placeholder={slot.placeholder ?? 'Model default'}
        size="xs"
        value={selectedValue}
        onChange={onChange}
      />
    </Field>
  );
};

export const GenerateComponentsSection = ({ onCommit, selectedModel, settings }: GenerateComponentsSectionProps) => {
  if (!selectedModel || selectedModel.type === 'external_image_generator') {
    return null;
  }

  const policy = getComponentSectionPolicy(selectedModel, settings);

  if (policy.slots.length === 0) {
    return null;
  }

  const ctx = getComponentPolicyContext(selectedModel, settings);
  const commitSlotValue = (slot: ComponentSlotPolicy, model: ModelConfig | null) => {
    if (slot.valueKind === 'main') {
      onCommit({ [slot.key]: isMainModelConfig(model) ? model : null });
      return;
    }

    if (slot.valueKind === 'vae') {
      onCommit({ [slot.key]: isVaeModelConfig(model) ? model : null });
      return;
    }

    onCommit({ [slot.key]: toComponentModel(model) });
  };

  return (
    <GenerateCollapsibleSection label="Components" defaultOpen={policy.defaultOpen}>
      <Stack gap="2" p="2">
        {policy.slots.map((slot) => (
          <ComponentPicker key={slot.key} ctx={ctx} slot={slot} onChange={(model) => commitSlotValue(slot, model)} />
        ))}
      </Stack>
    </GenerateCollapsibleSection>
  );
};
