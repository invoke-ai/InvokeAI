import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import type { GroupBase } from 'chakra-react-select';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

import { useGroupedModelCombobox } from './useGroupedModelCombobox';
import { useRelatedModelKeys } from './useRelatedModelKeys';
import { useSelectedModelKeys } from './useSelectedModelKeys';

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

export function useRelatedGroupedModelCombobox<T extends AnyModelConfig>({
  modelConfigs,
  selectedModel,
  onChange,
  isLoading = false,
  getIsDisabled,
  groupByType,
}: UseRelatedGroupedModelComboboxArg<T>): UseRelatedGroupedModelComboboxReturn {
  const { t } = useTranslation();

  const selectedKeys = useSelectedModelKeys();

  const relatedKeys = useRelatedModelKeys(selectedKeys);

  // Base grouped options
  const base = useGroupedModelCombobox({
    modelConfigs,
    selectedModel,
    onChange,
    getIsDisabled,
    isLoading,
    groupByType,
  });

  // If no related models selected, just return base
  if (relatedKeys.size === 0) {
    return base;
  }

  const relatedOptions: ComboboxOption[] = [];
  const updatedGroups: GroupBase<ComboboxOption>[] = [];

  for (const group of base.options) {
    const remainingOptions: ComboboxOption[] = [];

    for (const option of group.options) {
      if (relatedKeys.has(option.value)) {
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

  const finalOptions: GroupBase<ComboboxOption>[] =
    relatedOptions.length > 0
      ? [{ label: t('modelManager.relatedModels'), options: relatedOptions }, ...updatedGroups]
      : updatedGroups;

  return {
    ...base,
    options: finalOptions,
  };
}
