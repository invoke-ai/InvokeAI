import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { groupBy, reduce } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

type UseGroupedModelComboboxArg<T extends AnyModelConfig> = {
  modelConfigs: T[];
  selectedModel?: ModelIdentifierField | null;
  onChange: (value: T | null) => void;
  getIsDisabled?: (model: T) => boolean;
  isLoading?: boolean;
};

type UseGroupedModelComboboxReturn = {
  value: ComboboxOption | undefined | null;
  options: GroupBase<ComboboxOption>[];
  onChange: ComboboxOnChange;
  placeholder: string;
  noOptionsMessage: () => string;
};

export const useGroupedModelCombobox = <T extends AnyModelConfig>(
  arg: UseGroupedModelComboboxArg<T>
): UseGroupedModelComboboxReturn => {
  const { t } = useTranslation();
  const base_model = useAppSelector((s) => s.generation.model?.base ?? 'sdxl');
  const { modelConfigs, selectedModel, getIsDisabled, onChange, isLoading } = arg;
  const options = useMemo<GroupBase<ComboboxOption>[]>(() => {
    if (!modelConfigs) {
      return [];
    }
    const groupedModels = groupBy(modelConfigs, 'base');
    const _options = reduce(
      groupedModels,
      (acc, val, label) => {
        acc.push({
          label,
          options: val.map((model) => ({
            label: model.name,
            value: model.key,
            isDisabled: getIsDisabled ? getIsDisabled(model) : false,
          })),
        });
        return acc;
      },
      [] as GroupBase<ComboboxOption>[]
    );
    _options.sort((a) => (a.label === base_model ? -1 : 1));
    return _options;
  }, [getIsDisabled, modelConfigs, base_model]);

  const value = useMemo(
    () =>
      options.flatMap((o) => o.options).find((m) => (selectedModel ? m.value === selectedModel.key : false)) ?? null,
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
