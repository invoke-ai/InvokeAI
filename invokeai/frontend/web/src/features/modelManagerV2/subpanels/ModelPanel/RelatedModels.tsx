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

type ModelGroup = {
  type: string;
  label: string;
  color: string;
  models: AnyModelConfig[];
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

// Drying out and setting up for potential export

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

// Extracts model type from a label string (e.g., 'Base/LoRA' → 'lora')
const getTypeFromLabel = (label: string): string => label.split('/')[1]?.trim().toLowerCase() || '';

export const RelatedModels = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [addModelRelationship, { isLoading: isAdding }] = useAddModelRelationshipMutation();
  const [removeModelRelationship, { isLoading: isRemoving }] = useRemoveModelRelationshipMutation();
  const isLoading = isAdding || isRemoving;
  const [selectedKey, setSelectedKey] = useState('');
  const { data: modelConfigs } = useGetModelConfigsQuery();
  const { data: relatedModels = [] } = useGetRelatedModelIdsQuery(modelConfig.key);
  const relatedIDs = useMemo(() => new Set(relatedModels), [relatedModels]);

  // Defines model types to prioritize first in UI sorting.
  // Types not listed here will appear afterward in default order.
  const MODEL_TYPE_PRIORITY = useMemo(() => ['main', 'lora'], []);

  // Defines disallowed connection types.
  const DISALLOWED_RELATIONSHIPS = useMemo(
    () =>
      new Set([
        'main|main',
        'vae|vae',
        'controlnet|controlnet',
        'clip_vision|clip_vision',
        'control_lora|control_lora',
        'clip_embed|clip_embed',
        'spandrel_image_to_image|spandrel_image_to_image',
        'siglip|siglip',
        'flux_redux|flux_redux',
      ]),
    []
  );

  // Drying out sorting
  const prioritySort = useCallback(
    (a: string, b: string): number => {
      const aIndex = MODEL_TYPE_PRIORITY.indexOf(a);
      const bIndex = MODEL_TYPE_PRIORITY.indexOf(b);

      const aScore = aIndex === -1 ? 99 : aIndex;
      const bScore = bIndex === -1 ? 99 : bIndex;

      return aScore - bScore;
    },
    [MODEL_TYPE_PRIORITY]
  );

  //Get all modelConfigs that are not already related to the current model.
  const availableModels = useMemo(() => {
    if (!modelConfigs) {
      return [];
    }
    const isDisallowedRelationship = (a: string, b: string): boolean =>
      DISALLOWED_RELATIONSHIPS.has(`${a}|${b}`) || DISALLOWED_RELATIONSHIPS.has(`${b}|${a}`);

    return Object.values(modelConfigs.entities).filter(
      (m): m is AnyModelConfig =>
        !!m &&
        m.key !== modelConfig.key &&
        !relatedIDs.has(m.key) &&
        isBaseCompatible(modelConfig, m) &&
        !isDisallowedRelationship(modelConfig.type, m.type)
    );
  }, [modelConfigs, modelConfig, relatedIDs, DISALLOWED_RELATIONSHIPS]);

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
    await addModelRelationship({ model_key_1: modelConfig.key, model_key_2: target.key });
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

  // Finds the selected model's combobox option to control current dropdown state.
  const selectedOption = useMemo(() => {
    return options.flatMap((group) => group.options).find((o) => o.value === selectedKey) ?? null;
  }, [selectedKey, options]);

  const sortedOptions = useMemo(() => {
    return [...options].sort((a, b) => prioritySort(getTypeFromLabel(a.label ?? ''), getTypeFromLabel(b.label ?? '')));
  }, [options, prioritySort]);

  const groupedModelConfigs = useMemo(() => {
    if (!modelConfigs) {
      return [];
    }

    const models = [...relatedModels].map((id) => modelConfigs.entities[id]).filter((m): m is AnyModelConfig => !!m);

    models.sort((a, b) => prioritySort(a.type, b.type) || a.type.localeCompare(b.type) || a.name.localeCompare(b.name));

    const groupsMap = new Map<string, ModelGroup>();

    for (const model of models) {
      if (!groupsMap.has(model.type)) {
        groupsMap.set(model.type, {
          type: model.type,
          label: model.type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
          color: getModelTagColor(model.type),
          models: [],
        });
      }
      groupsMap.get(model.type)!.models.push(model);
    }

    return Array.from(groupsMap.values());
  }, [modelConfigs, relatedModels, prioritySort]);

  const removeHandlers = useMemo(() => {
    const map = new Map<string, () => void>();
    if (!modelConfigs) {
      return map;
    }

    for (const group of groupedModelConfigs) {
      for (const model of group.models) {
        map.set(model.key, () => {
          const target = modelConfigs.entities[model.key];
          if (!target) {
            return;
          }

          removeModelRelationship({
            model_key_1: modelConfig.key,
            model_key_2: model.key,
          }).unwrap();
        });
      }
    }

    return map;
  }, [groupedModelConfigs, modelConfig.key, modelConfigs, removeModelRelationship]);

  return (
    <Flex direction="column" gap="5" w="full">
      <FormLabel>{t('modelManager.relatedModels')}</FormLabel>
      <FormControl isInvalid={errors.length > 0}>
        <Flex gap="3" alignItems="center" w="full">
          <Combobox
            value={selectedOption}
            placeholder={placeholder}
            options={sortedOptions}
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
      {groupedModelConfigs.length > 0 && (
        <Box>
          <Flex gap="2" flexWrap="wrap">
            {groupedModelConfigs.map((group, i) => {
              const withDivider = i < groupedModelConfigs.length - 1;

              return (
                <Box key={group.type} mb={4}>
                  <ModelTagGroup group={group} isLoading={isLoading} removeHandlers={removeHandlers} />
                  {withDivider && <Divider my={4} opacity={0.3} />}
                </Box>
              );
            })}
          </Flex>
        </Box>
      )}
    </Flex>
  );
});

const ModelTag = ({
  model,
  onRemove,
  isLoading,
}: {
  model: AnyModelConfig;
  onRemove: () => void;
  isLoading: boolean;
}) => {
  return (
    <Tag py={2} px={4} bg={`${getModelTagColor(model.type)}.700`}>
      <Tooltip label={`${model.type}: ${model.name}`} hasArrow>
        <TagLabel maxWidth="50px" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
          {model.name}
        </TagLabel>
      </Tooltip>
      <TagCloseButton onClick={onRemove} isDisabled={isLoading} />
    </Tag>
  );
};

const ModelTagGroup = ({
  group,
  isLoading,
  removeHandlers,
}: {
  group: ModelGroup;
  isLoading: boolean;
  removeHandlers: Map<string, () => void>;
}) => {
  return (
    <Flex gap="2" flexWrap="wrap" alignItems="center">
      {group.models.map((model) => (
        <ModelTag key={model.key} model={model} onRemove={removeHandlers.get(model.key)!} isLoading={isLoading} />
      ))}
    </Flex>
  );
};

RelatedModels.displayName = 'RelatedModels';
