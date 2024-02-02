import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import { map } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfigEntity } from 'services/api/endpoints/models';
import { getModelId } from 'services/api/endpoints/models';

type UseModelComboboxArg<T extends AnyModelConfigEntity> = {
  modelEntities: EntityState<T, string> | undefined;
  selectedModel?: Pick<T, 'base_model' | 'model_name' | 'model_type'> | null;
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

export const useModelCombobox = <T extends AnyModelConfigEntity>(
  arg: UseModelComboboxArg<T>
): UseModelComboboxReturn => {
  const { t } = useTranslation();
  const { modelEntities, selectedModel, getIsDisabled, onChange, isLoading, optionsFilter = () => true } = arg;
  const options = useMemo<ComboboxOption[]>(() => {
    if (!modelEntities) {
      return [];
    }
    return map(modelEntities.entities)
      .filter(optionsFilter)
      .map((model) => ({
        label: model.model_name,
        value: model.id,
        isDisabled: getIsDisabled ? getIsDisabled(model) : false,
      }));
  }, [optionsFilter, getIsDisabled, modelEntities]);

  const value = useMemo(
    () => options.find((m) => (selectedModel ? m.value === getModelId(selectedModel) : false)),
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
