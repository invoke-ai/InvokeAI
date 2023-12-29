import type { EntityState } from '@reduxjs/toolkit';
import { map } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfigEntity } from 'services/api/endpoints/models';
import { getModelId } from 'services/api/endpoints/models';

import type { InvSelectOnChange, InvSelectOption } from './types';

type UseModelInvSelectArg<T extends AnyModelConfigEntity> = {
  modelEntities: EntityState<T, string> | undefined;
  selectedModel?: Pick<T, 'base_model' | 'model_name' | 'model_type'> | null;
  onChange: (value: T | null) => void;
  getIsDisabled?: (model: T) => boolean;
  optionsFilter?: (model: T) => boolean;
  isLoading?: boolean;
};

type UseModelInvSelectReturn = {
  value: InvSelectOption | undefined | null;
  options: InvSelectOption[];
  onChange: InvSelectOnChange;
  placeholder: string;
  noOptionsMessage: () => string;
};

export const useModelInvSelect = <T extends AnyModelConfigEntity>(
  arg: UseModelInvSelectArg<T>
): UseModelInvSelectReturn => {
  const { t } = useTranslation();
  const {
    modelEntities,
    selectedModel,
    getIsDisabled,
    onChange,
    isLoading,
    optionsFilter = () => true,
  } = arg;
  const options = useMemo<InvSelectOption[]>(() => {
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
    () =>
      options.find((m) =>
        selectedModel ? m.value === getModelId(selectedModel) : false
      ),
    [options, selectedModel]
  );

  const _onChange = useCallback<InvSelectOnChange>(
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
