import type { GenerateLora, GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

import { Badge, HStack, InputGroup, NumberInput, Slider, Stack, Switch, Text } from '@chakra-ui/react';
import { Field, IconButton, Tooltip } from '@workbench/components/ui';
import {
  DEFAULT_LORA_WEIGHT_CONFIG,
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  syncGenerateLorasWithModels,
} from '@workbench/generation/settings';
import { getModelBaseColorPalette, getModelBaseLabel } from '@workbench/models/baseIdentity';
import { ModelSelect } from '@workbench/models/components';
import { Trash2Icon } from 'lucide-react';
import { useMemo } from 'react';

import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';

interface GenerateConceptsSectionProps {
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const LORA_WEIGHT_MARKS = [
  { label: '-1', value: -1 },
  { label: '0', value: 0 },
  { label: '1', value: 1 },
  { label: '2', value: 2 },
];

const isCompatibleLora = (lora: GenerateLora, selectedModel: GenerateModelConfig | undefined): boolean =>
  Boolean(selectedModel && isLoraCompatibleWithModel(lora.model, selectedModel));

export const GenerateConceptsSection = ({
  loraModels,
  onCommit,
  selectedModel,
  settings,
}: GenerateConceptsSectionProps) => {
  const loras = useMemo(() => syncGenerateLorasWithModels(settings.loras, loraModels), [loraModels, settings.loras]);
  const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);
  const activeCount = selectedModel
    ? loras.filter((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, selectedModel)).length
    : 0;

  const addLora = (model: ModelConfig | null) => {
    if (!isLoraModelConfig(model) || selectedLoraKeys.has(model.key)) {
      return;
    }

    onCommit({
      loras: [...loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }],
    });
  };

  const updateLora = (modelKey: string, patch: Partial<Pick<GenerateLora, 'isEnabled' | 'weight'>>) => {
    onCommit({
      loras: loras.map((lora) => (lora.model.key === modelKey ? { ...lora, ...patch } : lora)),
    });
  };

  const removeLora = (modelKey: string) => {
    onCommit({ loras: loras.filter((lora) => lora.model.key !== modelKey) });
  };

  const badges =
    activeCount > 0 ? (
      <Badge colorPalette="green" size="xs" variant="surface">
        {activeCount} active
      </Badge>
    ) : loras.length > 0 ? (
      <Badge size="xs" variant="surface">
        {loras.length} off
      </Badge>
    ) : null;

  return (
    <GenerateCollapsibleSection label="Concepts" defaultOpen badges={badges}>
      <Stack gap="2" p="2">
        <Field label="Add concept" helpText={selectedModel ? undefined : 'Select a main model before adding concepts.'}>
          <ModelSelect
            excludeKeys={selectedLoraKeys}
            filter={(model) =>
              Boolean(selectedModel && isLoraModelConfig(model) && isLoraCompatibleWithModel(model, selectedModel))
            }
            modelTypes={['lora']}
            placeholder={selectedModel ? 'Search compatible concepts...' : 'Select a model first'}
            size="xs"
            value={null}
            onChange={addLora}
          />
        </Field>

        {loras.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs">
            Add LoRAs/concepts here to blend them into this generation.
          </Text>
        ) : (
          <Stack gap="2">
            {loras.map((lora) => (
              <LoraRow
                key={lora.model.key}
                isCompatible={isCompatibleLora(lora, selectedModel)}
                lora={lora}
                onRemove={() => removeLora(lora.model.key)}
                onToggle={(isEnabled) => updateLora(lora.model.key, { isEnabled })}
                onWeightChange={(weight) => updateLora(lora.model.key, { weight })}
              />
            ))}
          </Stack>
        )}
      </Stack>
    </GenerateCollapsibleSection>
  );
};

const LoraRow = ({
  isCompatible,
  lora,
  onRemove,
  onToggle,
  onWeightChange,
}: {
  isCompatible: boolean;
  lora: GenerateLora;
  onRemove: () => void;
  onToggle: (isEnabled: boolean) => void;
  onWeightChange: (weight: number) => void;
}) => {
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);

  return (
    <Stack
      bg="bg.subtle"
      borderColor={isActive ? 'border.emphasized' : 'border.subtle'}
      borderWidth="1px"
      gap="2"
      opacity={isCompatible ? 1 : 0.68}
      p="2"
      rounded="md"
    >
      <HStack gap="2" minW="0">
        <Stack flex="1" gap="0.5" minW="0">
          <HStack gap="1.5" minW="0">
            <Text color={isActive ? 'fg' : 'fg.muted'} fontSize="xs" fontWeight="medium" minW="0" truncate>
              {lora.model.name}
            </Text>
            <Badge colorPalette={getModelBaseColorPalette(lora.model.base)} flexShrink={0} size="xs" variant="surface">
              {getModelBaseLabel(lora.model.base)}
            </Badge>
            {!isCompatible ? (
              <Badge colorPalette="orange" flexShrink={0} size="xs" variant="surface">
                Incompatible
              </Badge>
            ) : null}
          </HStack>
          {lora.model.trigger_phrases?.length ? (
            <Text color="fg.subtle" fontSize="2xs" truncate>
              {lora.model.trigger_phrases.join(', ')}
            </Text>
          ) : null}
        </Stack>

        <HStack flexShrink="0" gap="1">
          <Switch.Root
            aria-label={`${isActive ? 'Disable' : 'Enable'} ${lora.model.name}`}
            checked={isActive}
            disabled={!isCompatible}
            size="sm"
            onCheckedChange={(event) => onToggle(event.checked)}
          >
            <Switch.HiddenInput />
            <Switch.Control _checked={{ bg: 'accent.solid' }}>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Root>

          <Tooltip content="Remove concept">
            <IconButton
              aria-label={`Remove ${lora.model.name}`}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={onRemove}
            >
              <Trash2Icon />
            </IconButton>
          </Tooltip>
        </HStack>
      </HStack>

      <HStack gap="2">
        <Slider.Root
          aria-label={[`${lora.model.name} weight`]}
          disabled={!isActive}
          flex="1"
          size="sm"
          max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
          min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
          minW="0"
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          value={[lora.weight]}
          onValueChange={({ value }) => {
            const nextWeight = value[0];

            if (Number.isFinite(nextWeight)) {
              onWeightChange(nextWeight as number);
            }
          }}
        >
          <Slider.Control>
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumbs />
          </Slider.Control>
          <Slider.Marks marks={LORA_WEIGHT_MARKS} />
        </Slider.Root>

        <NumberInput.Root
          max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
          min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
          size="xs"
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          value={String(lora.weight)}
          w="20"
          disabled={!isActive}
          onValueChange={({ valueAsNumber }) => {
            if (Number.isFinite(valueAsNumber)) {
              onWeightChange(valueAsNumber);
            }
          }}
        >
          <InputGroup
            endElement={
              <ModelDefaultButton
                disabled={!isActive || lora.weight === defaultWeight}
                label="Use concept default weight"
                onClick={() => onWeightChange(defaultWeight)}
              />
            }
            endElementProps={{ pointerEvents: 'auto' }}
          >
            <NumberInput.Input aria-label={`${lora.model.name} weight`} />
          </InputGroup>
        </NumberInput.Root>
      </HStack>
    </Stack>
  );
};
