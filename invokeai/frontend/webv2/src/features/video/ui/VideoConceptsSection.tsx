import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';

import { HStack, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { getDefaultLoraWeight, isLoraCompatibleWithModel, isLoraModelConfig } from '@features/generation/settings';
import { ModelSelect } from '@features/models/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Trash2Icon } from 'lucide-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoLorasEquivalent, areVideoModelsEquivalent } from './videoComparators';

/**
 * Concepts (LoRAs) for the Video panel. The picker is an adder (never shows a
 * selection); compatibility follows the shared generation rule, which for Wan
 * already encodes the A14B/5B family split. Expert routing is the graph's job
 * ('auto' reads each LoRA's probed high/low tag), so rows carry no target
 * control.
 */

const LORA_MODEL_TYPES: readonly ModelTaxonomyType[] = ['lora'];
const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

const VideoLoraRow = memo(function VideoLoraRow({
  lora,
  onRemove,
  onUpdate,
}: {
  lora: GenerateLora;
  onRemove: (key: string) => void;
  onUpdate: (key: string, update: Partial<GenerateLora>) => void;
}) {
  const { t } = useTranslation();
  const modelKey = lora.model.key;
  const handleToggle = useCallback(
    (details: { checked: boolean }) => onUpdate(modelKey, { isEnabled: details.checked }),
    [modelKey, onUpdate]
  );
  const handleWeightChange = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onUpdate(modelKey, { weight: valueAsNumber });
      }
    },
    [modelKey, onUpdate]
  );
  const handleRemove = useCallback(() => onRemove(modelKey), [modelKey, onRemove]);

  return (
    <HStack bg="bg.subtle" gap="2" p="2" rounded="md">
      <Switch.Root aria-label={lora.model.name} checked={lora.isEnabled} size="sm" onCheckedChange={handleToggle}>
        <Switch.HiddenInput />
        <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
          <Switch.Thumb />
        </Switch.Control>
      </Switch.Root>
      <MiddleTruncate flex="1" fontSize="xs" minW="0" text={lora.model.name} />
      <NumberInput.Root
        max={10}
        min={-10}
        size="xs"
        step={0.05}
        value={String(lora.weight)}
        w="20"
        onValueChange={handleWeightChange}
      >
        <NumberInput.Input aria-label={t('widgets.video.loraWeight', { name: lora.model.name })} />
      </NumberInput.Root>
      <IconButton
        aria-label={t('widgets.video.removeLora', { name: lora.model.name })}
        size="xs"
        variant="ghost"
        onClick={handleRemove}
      >
        <Trash2Icon />
      </IconButton>
    </HStack>
  );
});

export const VideoConceptsSection = memo(
  function VideoConceptsSection({
    loras,
    model,
    onChangeLoras,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    onChangeLoras: (loras: GenerateLora[]) => void;
  }) {
    const { t } = useTranslation();
    const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);
    const loraFilter = useCallback(
      (candidate: ModelConfig) =>
        Boolean(model && isLoraModelConfig(candidate) && isLoraCompatibleWithModel(candidate, model)),
      [model]
    );
    const addLora = useCallback(
      (candidate: ModelConfig | null) => {
        if (!model || !isLoraModelConfig(candidate) || !isLoraCompatibleWithModel(candidate, model)) {
          return;
        }

        onChangeLoras([...loras, { isEnabled: true, model: candidate, weight: getDefaultLoraWeight(candidate) }]);
      },
      [loras, model, onChangeLoras]
    );
    const updateLora = useCallback(
      (key: string, update: Partial<GenerateLora>) =>
        onChangeLoras(loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora))),
      [loras, onChangeLoras]
    );
    const removeLora = useCallback(
      (key: string) => onChangeLoras(loras.filter((candidate) => candidate.model.key !== key)),
      [loras, onChangeLoras]
    );

    return (
      <GenerationSettingsSection label={t('widgets.video.concepts')} sectionId="video-concepts" defaultOpen>
        <Stack gap="2" p="2">
          <ModelSelect
            disabled={!model}
            excludeKeys={selectedLoraKeys}
            filter={loraFilter}
            modelTypes={LORA_MODEL_TYPES}
            placeholder={t('widgets.video.addLora')}
            size="xs"
            value={null}
            onChange={addLora}
          />
          {loras.length === 0 ? (
            <Text color="fg.muted" fontSize="2xs">
              {t('widgets.video.noLoras')}
            </Text>
          ) : (
            loras.map((lora) => (
              <VideoLoraRow key={lora.model.key} lora={lora} onRemove={removeLora} onUpdate={updateLora} />
            ))
          )}
        </Stack>
      </GenerationSettingsSection>
    );
  },
  (previous, next) =>
    previous.onChangeLoras === next.onChangeLoras &&
    areVideoModelsEquivalent(previous.model, next.model) &&
    areVideoLorasEquivalent(previous.loras, next.loras)
);
