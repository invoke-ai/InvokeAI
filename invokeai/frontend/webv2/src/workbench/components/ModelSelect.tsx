import { Badge, Combobox, createListCollection, HStack, Portal, Stack, Text } from '@chakra-ui/react';
import { useEffect, useMemo, useState } from 'react';

import { ensureModelsLoaded, useModelsSnapshot } from '../models/modelsStore';
import { useWorkbenchPreferences } from '../settings/store';
import {
  formatBytes,
  getModelBaseColorPalette,
  getModelBaseLabel,
  getModelCategoryRank,
  getModelTypePluralLabel,
} from '../models/taxonomy';
import type { ModelConfig, ModelTaxonomyType } from '../models/types';

/**
 * Universal single-model picker: a searchable combobox over the installed
 * library. Scope it with `modelTypes` — one type for dedicated pickers (a
 * LoRA picker, a main-model picker) or several for cross-type pickers, in
 * which case results are grouped by type.
 */
export const ModelSelect = ({
  excludeKeys,
  filter,
  modelTypes,
  onChange,
  placeholder,
  size = 'sm',
  value,
}: {
  /** Hide specific models (e.g. the current model and already-linked ones). */
  excludeKeys?: ReadonlySet<string>;
  /** Extra predicate, e.g. base-architecture compatibility. */
  filter?: (model: ModelConfig) => boolean;
  /** The model types this instance searches. */
  modelTypes: ModelTaxonomyType[];
  onChange: (model: ModelConfig | null) => void;
  placeholder?: string;
  size?: 'xs' | 'sm' | 'md';
  /** Selected model key, or null. */
  value: string | null;
}) => {
  const { enableModelDescriptions } = useWorkbenchPreferences();
  const { models } = useModelsSnapshot();
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const candidates = useMemo(() => {
    const allowedTypes = new Set(modelTypes);

    return models.filter(
      (model) => allowedTypes.has(model.type) && !excludeKeys?.has(model.key) && (filter ? filter(model) : true)
    );
  }, [excludeKeys, filter, modelTypes, models]);

  const visibleItems = useMemo(() => {
    const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const matches =
      terms.length === 0
        ? candidates
        : candidates.filter((model) => {
            const haystack = `${model.name} ${model.base} ${model.type}`.toLowerCase();

            return terms.every((term) => haystack.includes(term));
          });

    return [...matches].sort(
      (a, b) =>
        getModelCategoryRank(a.type) - getModelCategoryRank(b.type) ||
        a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
    );
  }, [candidates, searchTerm]);

  /** Items grouped by type, preserving the canonical category order. */
  const groups = useMemo(() => {
    const byType = new Map<ModelTaxonomyType, ModelConfig[]>();

    for (const model of visibleItems) {
      const group = byType.get(model.type);

      if (group) {
        group.push(model);
      } else {
        byType.set(model.type, [model]);
      }
    }

    return [...byType.entries()];
  }, [visibleItems]);

  const collection = useMemo(
    () =>
      createListCollection({
        itemToString: (item: ModelConfig) => item.name,
        itemToValue: (item: ModelConfig) => item.key,
        items: visibleItems,
      }),
    [visibleItems]
  );

  const scopeLabel =
    modelTypes.length === 1 ? getModelTypePluralLabel(modelTypes[0] ?? 'main').toLowerCase() : 'models';

  return (
    <Combobox.Root
      collection={collection}
      openOnClick
      placeholder={placeholder ?? `Search ${scopeLabel}…`}
      selectionBehavior="replace"
      size={size}
      value={value === null ? [] : [value]}
      w="full"
      onInputValueChange={(details) => setSearchTerm(details.inputValue)}
      onValueChange={(details) => {
        onChange(details.items[0] ?? null);
      }}
    >
      <Combobox.Control>
        <Combobox.Input />
        <Combobox.IndicatorGroup>
          <Combobox.ClearTrigger />
          <Combobox.Trigger />
        </Combobox.IndicatorGroup>
      </Combobox.Control>
      <Portal>
        <Combobox.Positioner>
          <Combobox.Content
            bg="bg.surfaceRaised"
            borderColor="border.emphasis"
            borderWidth="1px"
            color="fg.default"
            maxH="18rem"
            rounded="lg"
            shadow="lg"
            zIndex="popover"
          >
            <Combobox.Empty color="fg.subtle" fontSize="2xs" p="2">
              {candidates.length === 0
                ? `No compatible ${scopeLabel} installed.`
                : `No ${scopeLabel} match your search.`}
            </Combobox.Empty>
            {groups.map(([type, groupModels]) => (
              <Combobox.ItemGroup key={type}>
                {modelTypes.length > 1 ? (
                  <Combobox.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                    {getModelTypePluralLabel(type)}
                  </Combobox.ItemGroupLabel>
                ) : null}
                {groupModels.map((model) => (
                  <Combobox.Item key={model.key} item={model}>
                    <HStack flex="1" gap="2" minW="0">
                      <Stack flex="1" gap="0" minW="0">
                        <Text fontSize="xs" minW="0" truncate>
                          {model.name}
                        </Text>
                        {enableModelDescriptions && model.description ? (
                          <Text color="fg.subtle" fontSize="2xs" truncate>
                            {model.description}
                          </Text>
                        ) : null}
                      </Stack>
                      <Badge
                        colorPalette={getModelBaseColorPalette(model.base)}
                        flexShrink={0}
                        fontSize="2xs"
                        size="sm"
                        variant="surface"
                      >
                        {getModelBaseLabel(model.base)}
                      </Badge>
                      <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
                        {formatBytes(model.file_size)}
                      </Text>
                    </HStack>
                    <Combobox.ItemIndicator />
                  </Combobox.Item>
                ))}
              </Combobox.ItemGroup>
            ))}
          </Combobox.Content>
        </Combobox.Positioner>
      </Portal>
    </Combobox.Root>
  );
};
