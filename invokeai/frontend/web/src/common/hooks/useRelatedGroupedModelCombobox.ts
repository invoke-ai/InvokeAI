import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { uniq } from 'lodash-es';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';
import type { AnyModelConfig } from 'services/api/types';

import { useGroupedModelCombobox } from './useGroupedModelCombobox';

type UseRelatedGroupedModelComboboxArg<T extends AnyModelConfig> = {
  modelConfigs: T[];
  selectedModel?: ModelIdentifierField | null;
  onChange: (value: T | null) => void;
  getIsDisabled?: (model: T) => boolean;
  isLoading?: boolean;
  groupByType?: boolean;
};

// Custom hook to overlay the grouped model combobox with related models on top!
// Cleaner than hooking into useGroupedModelCombobox with a flag to enable/disable the related models
// Also allows for related models to be shown conditionally with some pretty simple logic if it ends up as a config flag.

type UseRelatedGroupedModelComboboxReturn = {
  value: ComboboxOption | undefined | null;
  options: GroupBase<ComboboxOption>[];
  onChange: ComboboxOnChange;
  placeholder: string;
  noOptionsMessage: () => string;
};

const selectSelectedModelKeys = createMemoizedSelector(selectParamsSlice, selectLoRAsSlice, (params, loras) => {
  const keys: string[] = [];
  const main = params.model;
  const vae = params.vae;
  const refiner = params.refinerModel;
  const controlnet = params.controlLora;

  if (main) {
    keys.push(main.key);
  }
  if (vae) {
    keys.push(vae.key);
  }
  if (refiner) {
    keys.push(refiner.key);
  }
  if (controlnet) {
    keys.push(controlnet.key);
  }
  for (const { model } of loras.loras) {
    keys.push(model.key);
  }

  return uniq(keys);
});

export function useRelatedGroupedModelCombobox<T extends AnyModelConfig>({
  modelConfigs,
  selectedModel,
  onChange,
  isLoading = false,
  getIsDisabled,
  groupByType,
}: UseRelatedGroupedModelComboboxArg<T>): UseRelatedGroupedModelComboboxReturn {
  const { t } = useTranslation();

  const selectedKeys = useAppSelector(selectSelectedModelKeys);
  const { relatedKeys } = useGetRelatedModelIdsBatchQuery(selectedKeys, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { relatedKeys: EMPTY_ARRAY };
      }
      return { relatedKeys: data };
    },
  });

  // Base grouped options
  const base = useGroupedModelCombobox({
    modelConfigs,
    selectedModel,
    onChange,
    getIsDisabled,
    isLoading,
    groupByType,
  });

  const options = useMemo(() => {
    if (relatedKeys.length === 0) {
      return base.options;
    }

    const relatedOptions: ComboboxOption[] = [];
    const updatedGroups: GroupBase<ComboboxOption>[] = [];

    for (const group of base.options) {
      const remainingOptions: ComboboxOption[] = [];

      for (const option of group.options) {
        if (relatedKeys.includes(option.value)) {
          relatedOptions.push({ ...option, label: `* ${option.label}` });
        } else {
          remainingOptions.push(option);
        }
      }

      if (remainingOptions.length > 0) {
        updatedGroups.push({
          label: group.label,
          options: remainingOptions,
        });
      }
    }

    if (relatedOptions.length > 0) {
      return [{ label: t('modelManager.relatedModels'), options: relatedOptions }, ...updatedGroups];
    } else {
      return updatedGroups;
    }
  }, [base.options, relatedKeys, t]);

  return {
    ...base,
    options,
  };
}
