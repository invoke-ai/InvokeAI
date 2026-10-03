import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type {
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
} from '@features/generation/core/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import {
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  syncGenerateLorasWithModels,
} from '@features/generation/core/settings';
import { Field } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { useRegisterGenerateDraftFlusher } from './generateDraftRegistry';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { ConceptList, ConceptRow, type ConceptUpdate } from './shared/ConceptRow';
import { useDebouncedDraftValue } from './useDebouncedDraftValue';

interface GenerateConceptsContentProps {
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const LORA_WEIGHT_DEBOUNCE_MS = 250;

const isCompatibleLora = (lora: GenerateLora, selectedModel: GenerateModelConfig | undefined): boolean =>
  Boolean(selectedModel && isLoraCompatibleWithModel(lora.model, selectedModel));

export const GenerateConceptsContent = ({
  loraModels,
  onCommit,
  onCommitImmediate,
  projectId,
  selectedModel,
  settings,
}: GenerateConceptsContentProps) => {
  const { t } = useTranslation();
  const loras = useMemo(() => syncGenerateLorasWithModels(settings.loras, loraModels), [loraModels, settings.loras]);
  const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);

  const addLora = (model: ModelConfig | null) => {
    if (!isLoraModelConfig(model) || selectedLoraKeys.has(model.key)) {
      return;
    }

    onCommitImmediate({
      loras: [...loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }],
    });
  };

  const updateLora = (modelKey: string, patch: ConceptUpdate) => {
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

  return (
    <Stack gap="2">
      <Field
        hint="concepts"
        label={t('widgets.generate.addConcept')}
        helpText={selectedModel ? undefined : t('widgets.generate.selectMainModelBeforeConcepts')}
      >
        <ModelSelect
          excludeKeys={selectedLoraKeys}
          filter={(model) =>
            Boolean(selectedModel && isLoraModelConfig(model) && isLoraCompatibleWithModel(model, selectedModel))
          }
          modelTypes={['lora']}
          scopeLabel={t('models.scopeConcepts')}
          placeholder={
            selectedModel ? t('widgets.generate.searchCompatibleConcepts') : t('widgets.generate.selectModelFirst')
          }
          value={null}
          onChange={addLora}
        />
      </Field>

      {loras.length === 0 ? (
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.generate.addConceptsHelp')}
        </Text>
      ) : (
        <Box mx={-1}>
          <ConceptList label={t('widgets.generate.concepts')}>
            {loras.map((lora) => (
              <GenerateConceptRow
                key={lora.model.key}
                isCompatible={isCompatibleLora(lora, selectedModel)}
                lora={lora}
                projectId={projectId}
                onRemove={removeLora}
                onUpdate={updateLora}
              />
            ))}
          </ConceptList>
        </Box>
      )}
    </Stack>
  );
};

/** Holds the weight as a debounced draft so scrubbing does not commit settings per step. */
const GenerateConceptRow = ({
  isCompatible,
  lora,
  onRemove,
  onUpdate,
  projectId,
}: {
  isCompatible: boolean;
  lora: GenerateLora;
  onRemove: (key: string) => void;
  onUpdate: (key: string, update: ConceptUpdate) => void;
  projectId: string;
}) => {
  const models = useGenerationUi().models;
  const {
    draftValue: draftWeight,
    flushDraftValue,
    setDraftValue: setWeight,
  } = useDebouncedDraftValue({
    delayMs: LORA_WEIGHT_DEBOUNCE_MS,
    onCommit: (weight: number) => onUpdate(lora.model.key, { weight }),
    resetKey: projectId,
    value: lora.weight,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  return (
    <ConceptRow
      models={models}
      isCompatible={isCompatible}
      lora={draftWeight === lora.weight ? lora : { ...lora, weight: draftWeight }}
      onRemove={onRemove}
      onUpdate={(key, update) => (update.weight === undefined ? onUpdate(key, update) : setWeight(update.weight))}
    />
  );
};
