import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldControlNetModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { ControlNetModelFieldInputInstance, ControlNetModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';
import type { ControlNetModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ControlNetModelFieldInputInstance, ControlNetModelFieldInputTemplate>;

const ControlNetModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetControlNetModelsQuery();

  const _onChange = useCallback(
    (value: ControlNetModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldControlNetModelValueChanged({
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
    selectedModel: field.value,
    isLoading,
  });

  return (
    <Tooltip label={value?.description}>
      <FormControl className="nowheel nodrag" isInvalid={!value}>
        <Combobox
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Tooltip>
  );
};

export default memo(ControlNetModelFieldInputComponent);
