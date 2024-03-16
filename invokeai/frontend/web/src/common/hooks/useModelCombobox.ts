import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

type UseModelComboboxArg<T extends AnyModelConfig> = {
  modelConfigs: T[];
  selectedModel?: ModelIdentifierField | null;
  onChange: (value: T | null) => void;
  getIsDisabled?: (model: T) => boolean;
  optionsFilter?: (model: T) => boolean;
  isLoading?: boolean;
};

type UseModelComboboxReturn = {
  value: ComboboxOption | undefined | null;
  options: ComboboxOption[];
  onChange: ComboboxOnChange;
  placeholder: string;
  noOptionsMessage: () => string;
};

export const useModelCombobox = <T extends AnyModelConfig>(arg: UseModelComboboxArg<T>): UseModelComboboxReturn => {
  const { t } = useTranslation();
  const { modelConfigs, selectedModel, getIsDisabled, onChange, isLoading, optionsFilter = () => true } = arg;
  const options = useMemo<ComboboxOption[]>(() => {
    return modelConfigs.filter(optionsFilter).map((model) => ({
      label: model.name,
      value: model.key,
      isDisabled: getIsDisabled ? getIsDisabled(model) : false,
    }));
  }, [optionsFilter, getIsDisabled, modelConfigs]);

  const value = useMemo(
    () => options.find((m) => (selectedModel ? m.value === selectedModel.key : false)),
    [options, selectedModel]
  );

  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        onChange(null);
        return;
      }
      const model = modelConfigs.find((m) => m.key === v.value);
      if (!model) {
        onChange(null);
        return;
      }
      onChange(model);
    },
    [modelConfigs, onChange]
  );

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }

    if (options.length === 0) {
      return t('models.noModelsAvailable');
    }

    return t('models.selectModel');
  }, [isLoading, options, t]);

  const noOptionsMessage = useCallback(() => t('models.noMatchingModels'), [t]);

  return { options, value, onChange: _onChange, placeholder, noOptionsMessage };
};
