import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateLora, GenerateModelConfig, LoraModelConfig } from '@features/generation/core/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import { isLoraSupported } from '@features/generation/core/baseGenerationPolicies';
import {
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  syncGenerateLorasWithModels,
} from '@features/generation/core/settings';
import { Field } from '@platform/ui';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { ConceptList, ConceptRow, type ConceptUpdate } from './shared/ConceptRow';

interface GenerateConceptsContentProps {
  loraModels: LoraModelConfig[];
  loras: GenerateLora[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (update: GenerateSettingsUpdate) => void;
}

/** Whether the selected model's graph will load this concept: its family must take LoRAs and match the LoRA. */
export const isCompatibleLora = (lora: GenerateLora, selectedModel: GenerateModelConfig | undefined): boolean =>
  Boolean(selectedModel && isLoraSupported(selectedModel) && isLoraCompatibleWithModel(lora.model, selectedModel));

export const GenerateConceptsContent = ({
  loraModels,
  loras: draftLoras,
  onCommit,
  projectId,
  selectedModel,
}: GenerateConceptsContentProps) => {
  const { t } = useTranslation();
  const models = useGenerationUi().models;
  const loras = useMemo(() => syncGenerateLorasWithModels(draftLoras, loraModels), [draftLoras, loraModels]);
  const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);

  const addLora = (model: ModelConfig | null) => {
    if (!isLoraModelConfig(model)) {
      return;
    }

    onCommit((settings) => {
      const latestLoras = syncGenerateLorasWithModels(settings.loras, loraModels);

      if (latestLoras.some((lora) => lora.model.key === model.key)) {
        return settings;
      }

      return {
        ...settings,
        loras: [...latestLoras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }],
      };
    });
  };

  const updateLora = useCallback(
    (modelKey: string, patch: ConceptUpdate) => {
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
    },
    [loraModels, onCommit]
  );

  const removeLora = useCallback(
    (modelKey: string) => {
      onCommit((settings) => ({ ...settings, loras: settings.loras.filter((lora) => lora.model.key !== modelKey) }));
    },
    [onCommit]
  );

  const rows =
    loras.length > 0 ? (
      <Box mx={-1}>
        <ConceptList label={t('widgets.generate.concepts')} projectId={projectId}>
          {loras.map((lora) => (
            <ConceptRow
              key={lora.model.key}
              isCompatible={isCompatibleLora(lora, selectedModel)}
              lora={lora}
              models={models}
              onRemove={removeLora}
              onUpdate={updateLora}
            />
          ))}
        </ConceptList>
      </Box>
    ) : null;

  if (selectedModel && !isLoraSupported(selectedModel)) {
    // Rows left from another model stay listed, marked incompatible, so they can still be removed.
    return (
      <Stack gap="2">
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.generate.conceptsUnsupported')}
        </Text>
        {rows}
      </Stack>
    );
  }

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

      {rows ?? (
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.generate.addConceptsHelp')}
        </Text>
      )}
    </Stack>
  );
};
