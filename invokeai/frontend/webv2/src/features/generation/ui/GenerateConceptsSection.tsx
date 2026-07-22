import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type {
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
} from '@features/generation/core/types';

import { Badge, HStack, Stack, Switch, Text } from '@chakra-ui/react';
import {
  DEFAULT_LORA_WEIGHT_CONFIG,
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  syncGenerateLorasWithModels,
} from '@features/generation/core/settings';
import { Field, IconButton, Tooltip } from '@platform/ui';
import { Trash2Icon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { useRegisterGenerateDraftFlusher } from './generateDraftRegistry';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { SliderNumberField } from './shared/SliderNumberField';
import { useDebouncedDraftValue } from './useDebouncedDraftValue';

interface GenerateConceptsSectionProps {
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const LORA_WEIGHT_MARKS = [
  { label: '-1', value: -1 },
  { label: '0', value: 0 },
  { label: '1', value: 1 },
  { label: '2', value: 2 },
];
const LORA_WEIGHT_DEBOUNCE_MS = 250;

const isCompatibleLora = (lora: GenerateLora, selectedModel: GenerateModelConfig | undefined): boolean =>
  Boolean(selectedModel && isLoraCompatibleWithModel(lora.model, selectedModel));

export const GenerateConceptsSection = ({
  loraModels,
  onCommit,
  onCommitImmediate,
  projectId,
  selectedModel,
  settings,
}: GenerateConceptsSectionProps) => {
  const { t } = useTranslation();
  const loras = useMemo(() => syncGenerateLorasWithModels(settings.loras, loraModels), [loraModels, settings.loras]);
  const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);

  const activeCount = selectedModel
    ? loras.filter((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, selectedModel)).length
    : 0;

  const addLora = (model: ModelConfig | null) => {
    if (!isLoraModelConfig(model) || selectedLoraKeys.has(model.key)) {
      return;
    }

    onCommitImmediate({
      loras: [...loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }],
    });
  };

  const updateLora = (modelKey: string, patch: Partial<Pick<GenerateLora, 'isEnabled' | 'weight'>>) => {
    onCommit((settings) => {
      const latestLoras = syncGenerateLorasWithModels(settings.loras, loraModels);
      const hasLora = latestLoras.some((lora) => lora.model.key === modelKey);

      if (!hasLora) {
        return settings;
      }

      return {
        ...settings,
        loras: latestLoras.map((lora) => (lora.model.key === modelKey ? { ...lora, ...patch } : lora)),
      };
    });
  };

  const removeLora = (modelKey: string) => {
    onCommitImmediate({ loras: loras.filter((lora) => lora.model.key !== modelKey) });
  };

  const badges =
    activeCount > 0 ? (
      <Badge size="xs" variant="surface">
        {t('widgets.generate.activeCount', { count: activeCount })}
      </Badge>
    ) : loras.length > 0 ? (
      <Badge size="xs" variant="surface">
        {t('widgets.generate.offCount', { count: loras.length })}
      </Badge>
    ) : null;

  return (
    <GenerateCollapsibleSection label={t('widgets.generate.concepts')} defaultOpen badges={badges} sectionId="concepts">
      <Stack gap="2" p="2">
        <Field
          label={t('widgets.generate.addConcept')}
          helpText={selectedModel ? undefined : t('widgets.generate.selectMainModelBeforeConcepts')}
        >
          <ModelSelect
            excludeKeys={selectedLoraKeys}
            filter={(model) =>
              Boolean(selectedModel && isLoraModelConfig(model) && isLoraCompatibleWithModel(model, selectedModel))
            }
            modelTypes={['lora']}
            placeholder={
              selectedModel ? t('widgets.generate.searchCompatibleConcepts') : t('widgets.generate.selectModelFirst')
            }
            size="xs"
            value={null}
            onChange={addLora}
          />
        </Field>

        {loras.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.generate.addConceptsHelp')}
          </Text>
        ) : (
          <Stack gap="2">
            {loras.map((lora) => (
              <LoraRow
                key={lora.model.key}
                isCompatible={isCompatibleLora(lora, selectedModel)}
                lora={lora}
                projectId={projectId}
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
  projectId,
  onToggle,
  onWeightChange,
}: {
  isCompatible: boolean;
  lora: GenerateLora;
  onRemove: () => void;
  projectId: string;
  onToggle: (isEnabled: boolean) => void;
  onWeightChange: (weight: number) => void;
}) => {
  const { t } = useTranslation();
  const { getBaseColorPalette, getBaseLabel } = useGenerationUi().models;
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);

  const {
    draftValue: draftWeight,
    flushDraftValue,
    setDraftValue: setWeight,
  } = useDebouncedDraftValue({
    delayMs: LORA_WEIGHT_DEBOUNCE_MS,
    onCommit: onWeightChange,
    resetKey: projectId,
    value: lora.weight,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

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
            <Badge colorPalette={getBaseColorPalette(lora.model.base)} flexShrink={0} size="xs" variant="surface">
              {getBaseLabel(lora.model.base)}
            </Badge>
            {!isCompatible ? (
              <Badge colorPalette="orange" flexShrink={0} size="xs" variant="surface">
                {t('widgets.generate.incompatible')}
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
            aria-label={
              isActive
                ? t('widgets.generate.disableConcept', { name: lora.model.name })
                : t('widgets.generate.enableConcept', { name: lora.model.name })
            }
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

          <Tooltip content={t('widgets.generate.removeConcept')}>
            <IconButton
              aria-label={t('widgets.generate.removeConceptNamed', { name: lora.model.name })}
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

      <SliderNumberField
        ariaLabel={t('widgets.generate.conceptWeight', { name: lora.model.name })}
        defaultValue={defaultWeight}
        disabled={!isActive}
        marks={LORA_WEIGHT_MARKS}
        max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
        min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
        numberInputMax={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
        numberInputMin={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
        resetLabel={t('widgets.generate.useConceptDefaultWeight')}
        step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
        value={draftWeight}
        onChange={setWeight}
      />
    </Stack>
  );
};
