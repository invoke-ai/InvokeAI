import type { Item } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from "app/store/constants";
import type { ModelIdentifierWithBase } from 'features/nodes/types/common';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { filter } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

type UseModelCustomSelectArg<T extends AnyModelConfig> = {
  data: EntityState<T, string> | undefined;
  isLoading: boolean;
  selectedModel?: ModelIdentifierWithBase | null;
  onChange: (value: T | null) => void;
  modelFilter?: (model: T) => boolean;
  isModelDisabled?: (model: T) => boolean;
};

type UseModelCustomSelectReturn = {
  selectedItem: Item | null;
  items: Item[];
  onChange: (item: Item | null) => void;
  placeholder: string;
};

const modelFilterDefault = () => true;
const isModelDisabledDefault = () => false;

export const useModelCustomSelect = <T extends AnyModelConfig>({
  data,
  isLoading,
  selectedModel,
  onChange,
  modelFilter = modelFilterDefault,
  isModelDisabled = isModelDisabledDefault,
}: UseModelCustomSelectArg<T>): UseModelCustomSelectReturn => {
  const { t } = useTranslation();

  const items: Item[] = useMemo(
    () =>
      data
        ? filter(data.entities, modelFilter).map<Item>((m) => ({
            label: m.name,
            value: m.key,
            description: m.description,
            group: MODEL_TYPE_SHORT_MAP[m.base],
            isDisabled: isModelDisabled(m),
          }))
        : EMPTY_ARRAY,
    [data, isModelDisabled, modelFilter]
  );

  const _onChange = useCallback(
    (item: Item | null) => {
      if (!item || !data) {
        return;
      }
      const model = data.entities[item.value];
      if (!model) {
        return;
      }
      onChange(model);
    },
    [data, onChange]
  );

  const selectedItem = useMemo(() => items.find((o) => o.value === selectedModel?.key) ?? null, [selectedModel, items]);

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }

    if (items.length === 0) {
      return t('models.noModelsAvailable');
    }

    return t('models.selectModel');
  }, [isLoading, items, t]);

  return {
    items,
    onChange: _onChange,
    selectedItem,
    placeholder,
  };
};
