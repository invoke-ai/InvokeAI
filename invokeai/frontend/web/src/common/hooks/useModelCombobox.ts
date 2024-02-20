import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import type { ModelIdentifierWithBase } from 'features/nodes/types/common';
import { map } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

type UseModelComboboxArg<T extends AnyModelConfig> = {
  modelEntities: EntityState<T, string> | undefined;
  selectedModel?: ModelIdentifierWithBase | null;
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
  const { modelEntities, selectedModel, getIsDisabled, onChange, isLoading, optionsFilter = () => true } = arg;
  const options = useMemo<ComboboxOption[]>(() => {
    if (!modelEntities) {
      return [];
    }
    return map(modelEntities.entities)
      .filter(optionsFilter)
      .map((model) => ({
        label: model.name,
        value: model.key,
        isDisabled: getIsDisabled ? getIsDisabled(model) : false,
      }));
  }, [optionsFilter, getIsDisabled, modelEntities]);

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
      const model = modelEntities?.entities[v.value];
      if (!model) {
        onChange(null);
        return;
      }
      onChange(model);
    },
    [modelEntities?.entities, onChange]
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
