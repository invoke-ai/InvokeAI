import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { LoRAModelFieldInputInstance, LoRAModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<LoRAModelFieldInputInstance, LoRAModelFieldInputTemplate>;

const LoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const _onChange = useCallback(
    (value: LoRAModelConfig | null) => {
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
    modelConfigs,
    onChange: _onChange,
    selectedModel: field.value,
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
