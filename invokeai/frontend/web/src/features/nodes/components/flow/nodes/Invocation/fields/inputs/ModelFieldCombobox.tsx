import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { typedMemo } from 'common/util/typedMemo';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { AnyModelConfig } from 'services/api/types';

type Props<T extends AnyModelConfig> = {
  value: ModelIdentifierField | undefined;
  modelConfigs: T[];
  isLoadingConfigs: boolean;
  onChange: (value: T | null) => void;
  required: boolean;
  groupByType?: boolean;
};

const ModelFieldComboboxInternal = <T extends AnyModelConfig>({
  value: _value,
  modelConfigs,
  isLoadingConfigs,
  onChange: _onChange,
  required,
  groupByType,
}: Props<T>) => {
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    isLoading: isLoadingConfigs,
    selectedModel: _value,
    groupByType,
  });

  return (
    <FormControl
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      isDisabled={!options.length}
      isInvalid={!value && required}
      gap={2}
    >
      <Combobox
        value={value}
        placeholder={required ? placeholder : `(Optional) ${placeholder}`}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};

export const ModelFieldCombobox = typedMemo(ModelFieldComboboxInternal);
