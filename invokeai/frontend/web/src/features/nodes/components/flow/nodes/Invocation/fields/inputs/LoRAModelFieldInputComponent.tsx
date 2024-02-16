import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { LoRAModelFieldInputInstance, LoRAModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { LoRAConfig } from 'services/api/endpoints/models';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<LoRAModelFieldInputInstance, LoRAModelFieldInputTemplate>;

const LoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetLoRAModelsQuery();
  const _onChange = useCallback(
    (value: LoRAConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldLoRAModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelEntities: data,
    onChange: _onChange,
    selectedModel: field.value ? { ...field.value, model_type: 'lora' } : undefined,
    isLoading,
  });

  return (
    <FormControl className="nowheel nodrag" isInvalid={!value} isDisabled={!options.length}>
      <Combobox
        value={value}
        placeholder={placeholder}
        noOptionsMessage={noOptionsMessage}
        options={options}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(LoRAModelFieldInputComponent);
