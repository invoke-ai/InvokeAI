import { Checkbox, Flex, IconButton, Input, InputGroup, InputRightElement, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  type FilterableModelType,
  modelSelectionChanged,
  selectFilteredModelType,
  selectSearchTerm,
  selectSelectedModelKeys,
  setSearchTerm,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { t } from 'i18next';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { PiXBold } from 'react-icons/pi';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import { ModelTypeFilter } from './ModelTypeFilter';

export const ModelListNavigation = memo(() => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectSearchTerm);
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const selectedModelKeys = useAppSelector(selectSelectedModelKeys);
  const { data } = useGetModelConfigsQuery();

  // Calculate displayed (filtered) model keys
  const displayedModelKeys = useMemo(() => {
    const modelConfigs = modelConfigsAdapterSelectors.selectAll(data ?? { ids: [], entities: {} });
    const filteredModels = modelsFilter(modelConfigs, searchTerm, filteredModelType);
    return filteredModels.map((m) => m.key);
  }, [data, searchTerm, filteredModelType]);

  // Calculate checkbox state
  const { allSelected, someSelected } = useMemo(() => {
    if (displayedModelKeys.length === 0) {
      return { allSelected: false, someSelected: false };
    }
    const selectedSet = new Set(selectedModelKeys);
    const displayedSelectedCount = displayedModelKeys.filter((key) => selectedSet.has(key)).length;
    return {
      allSelected: displayedSelectedCount === displayedModelKeys.length,
      someSelected: displayedSelectedCount > 0 && displayedSelectedCount < displayedModelKeys.length,
    };
  }, [displayedModelKeys, selectedModelKeys]);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback(
    (event) => {
      dispatch(setSearchTerm(event.target.value));
    },
    [dispatch]
  );

  const clearSearch = useCallback(() => {
    dispatch(setSearchTerm(''));
  }, [dispatch]);

  const handleToggleAll = useCallback(() => {
    if (allSelected) {
      // Deselect all displayed models
      const displayedSet = new Set(displayedModelKeys);
      const newSelection = selectedModelKeys.filter((key) => !displayedSet.has(key));
      dispatch(modelSelectionChanged(newSelection));
    } else {
      // Select all displayed models (merge with existing selection)
      const selectedSet = new Set(selectedModelKeys);
      displayedModelKeys.forEach((key) => selectedSet.add(key));
      dispatch(modelSelectionChanged(Array.from(selectedSet)));
    }
  }, [allSelected, displayedModelKeys, selectedModelKeys, dispatch]);

  return (
    <Flex gap={2} alignItems="center" justifyContent="space-between">
      <Flex gap={2} alignItems="center">
        <Flex gap={2} alignItems="center" flexShrink={0}>
          <Checkbox
            isChecked={allSelected}
            isIndeterminate={someSelected}
            onChange={handleToggleAll}
            isDisabled={displayedModelKeys.length === 0}
            aria-label={t('modelManager.selectAll')}
          />
          <Text fontSize="sm" fontWeight="medium" whiteSpace="nowrap">
            {t('modelManager.selectAll')}
          </Text>
        </Flex>
        <InputGroup>
          <Input
            placeholder={t('modelManager.search')}
            value={searchTerm || ''}
            data-testid="board-search-input"
            onChange={handleSearch}
          />

          {!!searchTerm?.length && (
            <InputRightElement h="full" pe={2}>
              <IconButton
                size="sm"
                variant="link"
                aria-label={t('boards.clearSearch')}
                icon={<PiXBold />}
                onClick={clearSearch}
              />
            </InputRightElement>
          )}
        </InputGroup>
      </Flex>
      <Flex shrink={0}>
        <ModelTypeFilter />
      </Flex>
    </Flex>
  );
});

ModelListNavigation.displayName = 'ModelListNavigation';

const modelsFilter = <T extends AnyModelConfig>(
  data: T[],
  nameFilter: string,
  filteredModelType: FilterableModelType | null
): T[] => {
  return data.filter((model) => {
    const matchesFilter =
      model.name.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.base.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.type.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.description?.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.format.toLowerCase().includes(nameFilter.toLowerCase());

    const matchesType = getMatchesType(model, filteredModelType);

    return matchesFilter && matchesType;
  });
};

const getMatchesType = (modelConfig: AnyModelConfig, filteredModelType: FilterableModelType | null): boolean => {
  if (filteredModelType === 'refiner') {
    return modelConfig.base === 'sdxl-refiner';
  }

  if (filteredModelType === 'main' && modelConfig.base === 'sdxl-refiner') {
    return false;
  }

  return filteredModelType ? modelConfig.type === filteredModelType : true;
};
