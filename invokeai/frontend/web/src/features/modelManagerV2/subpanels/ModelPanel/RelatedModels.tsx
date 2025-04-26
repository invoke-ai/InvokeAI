/**
 * RelatedModels.tsx
 *
 * Panel for managing and displaying model-to-model relationships.
 *
 * Allows adding/removing bidirectional links between models, organized visually
 * with color-coded tags, dividers between types, and sorted dropdown selection.
 */

import {
  Box,
  Button,
  Combobox,
  Divider,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Tag,
  TagCloseButton,
  TagLabel,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import {
  useAddModelRelationshipMutation,
  useGetRelatedModelIdsQuery,
  useRemoveModelRelationshipMutation,
} from 'services/api/endpoints/modelRelationships';
import { useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  modelConfig: AnyModelConfig;
};

// Determines if two models are compatible for relationship linking based on their base type.
//
// Models with a base of 'any' are considered universally compatible.
// This is a known flaw: 'any'-based links may allow relationships that are
// meaningless in practice and could bloat the database over time.
//
// TODO: In the future, refine this logic to more strictly validate
// relationships based on model types or actual usage patterns.
const isBaseCompatible = (a: AnyModelConfig, b: AnyModelConfig): boolean => {
  if (a.base === 'any' || b.base === 'any') {
    return true;
  }
  return a.base === b.base;
};

export const RelatedModels = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [addModelRelationship, { isLoading: isAdding }] = useAddModelRelationshipMutation();
  const [removeModelRelationship, { isLoading: isRemoving }] = useRemoveModelRelationshipMutation();
  const isLoading = isAdding || isRemoving;
  const [selectedKey, setSelectedKey] = useState('');
  const { data: modelConfigs } = useGetModelConfigsQuery();
  const { data: relatedModels = [] } = useGetRelatedModelIdsQuery(modelConfig.key);
  const relatedIDs = useMemo(() => new Set(relatedModels), [relatedModels]);
  // Used to prioritize certain model types in UI sorting
  const MODEL_TYPE_PRIORITY = useMemo(() => ['main', 'lora'], []);

  //Get all modelConfigs that are not already related to the current model.
  const availableModels = useMemo(() => {
    if (!modelConfigs) {
      return [];
    }

    return Object.values(modelConfigs.entities).filter(
      (m): m is AnyModelConfig =>
        !!m &&
        m.key !== modelConfig.key &&
        !relatedIDs.has(m.key) &&
        isBaseCompatible(modelConfig, m) &&
        !(modelConfig.type === 'main' && m.type === 'main') // still block main↔main
    );
  }, [modelConfigs, modelConfig, relatedIDs]);

  // Tracks validation errors for current input (e.g., duplicate key or no selection).
  const errors = useMemo(() => {
    const errs: string[] = [];
    if (!selectedKey) {
      return errs;
    }
    if (relatedIDs.has(selectedKey)) {
      errs.push('Item already promoted');
    }
    return errs;
  }, [selectedKey, relatedIDs]);

  // Handles linking a selected model to the current one via API.
  const handleAdd = useCallback(async () => {
    const target = availableModels.find((m) => m.key === selectedKey);
    if (!target) {
      return;
    }

    setSelectedKey('');
    await Promise.all([addModelRelationship({ model_key_1: modelConfig.key, model_key_2: target.key })]);
  }, [modelConfig, availableModels, addModelRelationship, selectedKey]);

  const {
    options,
    onChange: comboboxOnChange,
    placeholder,
    noOptionsMessage,
  } = useGroupedModelCombobox({
    modelConfigs: availableModels,
    selectedModel: null,
    onChange: (model) => {
      if (!model) {
        return;
      }
      setSelectedKey(model.key);
    },
    groupByType: true,
  });

  // Unlinks an existing related model via API.
  const handleRemove = useCallback(
    async (id: string) => {
      const target = modelConfigs?.entities[id];
      if (!target) {
        return;
      }

      await Promise.all([removeModelRelationship({ model_key_1: modelConfig.key, model_key_2: target.key })]);
    },
    [modelConfig, modelConfigs, removeModelRelationship]
  );

  // Finds the selected model's combobox option to control current dropdown state.
  const selectedOption = useMemo(() => {
    return options.flatMap((group) => group.options).find((o) => o.value === selectedKey) ?? null;
  }, [selectedKey, options]);

  const makeRemoveHandler = useCallback((id: string) => () => handleRemove(id), [handleRemove]);

  // Defines custom tag colors for model types in the UI.
  //
  // The default UI color scheme (mostly grey and orange) felt too flat,
  // so this mapping provides a slightly more expressive color flow.
  //
  // Note: This is purely aesthetic. Safe to remove if project preferences change.
  const getModelTagColor = (type: string): string => {
    switch (type) {
      case 'main':
      case 'checkpoint':
        return 'orange';
      case 'lora':
      case 'lycoris':
        return 'purple';
      case 'embedding':
      case 'embedding_file':
        return 'teal';
      case 'vae':
        return 'blue';
      case 'controlnet':
      case 'ip_adapter':
      case 't2i_adapter':
        return 'cyan';
      case 'onnx':
      case 'bnb_quantized_int8b':
      case 'bnb_quantized_nf4b':
      case 'gguf_quantized':
        return 'pink';
      case 't5_encoder':
      case 'clip_embed':
      case 'clip_vision':
      case 'siglip':
        return 'green';
      default:
        return 'base';
    }
  };

  // Force group priority order: Main first, then LoRA
  const getTypeFromLabel = (label: string): string => label.split('/')[1]?.trim().toLowerCase() || '';

  const sortedOptions = useMemo(() => {
    return [...options].sort((a, b) => {
      const aType = getTypeFromLabel(a.label ?? '');
      const bType = getTypeFromLabel(b.label ?? '');

      const aIndex = MODEL_TYPE_PRIORITY.indexOf(aType);
      const bIndex = MODEL_TYPE_PRIORITY.indexOf(bType);

      const aScore = aIndex === -1 ? 99 : aIndex;
      const bScore = bIndex === -1 ? 99 : bIndex;

      return aScore - bScore;
    });
  }, [options, MODEL_TYPE_PRIORITY]);

  return (
    <Flex direction="column" gap="5" w="full">
      <FormLabel>{t('modelManager.relatedModels')}</FormLabel>
      <FormControl isInvalid={errors.length > 0}>
        <Flex gap="3" alignItems="center" w="full">
          <Combobox
            value={selectedOption}
            placeholder={placeholder}
            options={sortedOptions} // Sorts options to prioritize 'main' and 'lora' types at the top.
            onChange={comboboxOnChange}
            noOptionsMessage={noOptionsMessage}
          />
          <Button
            leftIcon={<PiPlusBold />}
            size="sm"
            onClick={handleAdd}
            isDisabled={!selectedKey || errors.length > 0}
            isLoading={isLoading}
          >
            {t('common.add')}
          </Button>
        </Flex>
        {errors.map((error) => (
          <FormErrorMessage key={error}>{error}</FormErrorMessage>
        ))}
      </FormControl>
      <Box>
        <Flex gap="2" flexWrap="wrap">
          {
            // Render the related model tags as styled components.
            //
            // Models are grouped visually by type, sorted with 'main' and 'lora' types at the front.
            // A vertical Divider is inserted when the type changes between adjacent models.
            // Tags include:
            //   - Colored background based on model type (via getModelTagColor)
            //   - Tooltip showing "<ModelType>: <ModelName>"
            //   - Ellipsis-truncated tag name for compact layout
            //   - A close button to remove the relationship
            [...relatedModels]
              .sort((aKey, bKey) => {
                const a = modelConfigs?.entities[aKey];
                const b = modelConfigs?.entities[bKey];
                if (!a || !b) {
                  return 0;
                }

                // Floats Mains and LoRAs to the front
                const aPriority = MODEL_TYPE_PRIORITY.indexOf(a.type);
                const bPriority = MODEL_TYPE_PRIORITY.indexOf(b.type);

                const aScore = aPriority === -1 ? 99 : aPriority;
                const bScore = bPriority === -1 ? 99 : bPriority;

                return aScore - bScore || a.type.localeCompare(b.type) || a.name.localeCompare(b.name);
              })
              .reduce<JSX.Element[]>((acc, id, index, arr) => {
                const model = modelConfigs?.entities[id];
                if (!model) {
                  return acc;
                }

                const modelName = model.name ?? id;
                const modelType = model.type ?? 'unknown';
                const modelTypeLabel = modelType.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

                //  Create a divider if the previous model is of a different type. Just a small dash of visual flair.
                const prevId = index > 0 ? arr[index - 1] : undefined;
                const prevModel = prevId ? modelConfigs?.entities[prevId] : null;
                const needsDivider = prevModel && prevModel.type !== model.type;

                if (needsDivider) {
                  acc.push(<Divider orientation="vertical" key={`divider-${id}`} opacity={0.3} />);
                }

                acc.push(
                  <Tag key={id} py={2} px={4} bg={`${getModelTagColor(model.type)}.700`}>
                    <Tooltip label={`${modelTypeLabel}: ${modelName}`} hasArrow>
                      <TagLabel
                        style={{
                          maxWidth: '50px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {modelName}
                      </TagLabel>
                    </Tooltip>
                    <TagCloseButton onClick={makeRemoveHandler(id)} isDisabled={isLoading} />
                  </Tag>
                );

                return acc;
              }, [])
          }
        </Flex>
      </Box>
    </Flex>
  );
});

RelatedModels.displayName = 'RelatedModels';
