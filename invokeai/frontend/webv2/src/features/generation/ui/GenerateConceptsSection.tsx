import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateLora, GenerateModelConfig, LoraModelConfig } from '@features/generation/core/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import { isLoraSupported } from '@features/generation/core/baseGenerationPolicies';
import { addConcept, type ConceptUpdate, removeConcept, updateConcept } from '@features/generation/core/conceptList';
import {
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  syncGenerateLorasWithModels,
} from '@features/generation/core/settings';
import { Field } from '@platform/ui';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { GenerationModelSelect as ModelSelect, useGenerationUi } from './GenerationUiContext';
import { ConceptList, ConceptRow } from './shared/ConceptRow';

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

  /** Edits the latest settings' list as the rows show it; an edit that changes nothing leaves the settings untouched. */
  const commitLoras = useCallback(
    (edit: (loras: GenerateLora[]) => GenerateLora[]) => {
      onCommit((settings) => {
        const latest = syncGenerateLorasWithModels(settings.loras, loraModels);
        const next = edit(latest);

        return next === latest ? settings : { ...settings, loras: next };
      });
    },
    [loraModels, onCommit]
  );

  const addLora = (model: ModelConfig | null) => {
    if (selectedModel && isLoraSupported(selectedModel) && isLoraModelConfig(model)) {
      commitLoras((latest) => addConcept(latest, model, selectedModel));
    }
  };

  const updateLora = useCallback(
    (modelKey: string, update: ConceptUpdate) => commitLoras((latest) => updateConcept(latest, modelKey, update)),
    [commitLoras]
  );

  const removeLora = useCallback(
    (modelKey: string) => commitLoras((latest) => removeConcept(latest, modelKey)),
    [commitLoras]
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
