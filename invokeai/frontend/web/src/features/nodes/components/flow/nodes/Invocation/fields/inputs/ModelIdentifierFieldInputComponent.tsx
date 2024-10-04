import { Combobox, Flex, FormControl } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldModelIdentifierValueChanged } from 'features/nodes/store/nodesSlice';
import type { ModelIdentifierFieldInputInstance, ModelIdentifierFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ModelIdentifierFieldInputInstance, ModelIdentifierFieldInputTemplate>;

const ModelIdentifierFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetModelConfigsQuery();
  const _onChange = useCallback(
    (value: AnyModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldModelIdentifierValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const modelConfigs = useMemo(() => {
    if (!data) {
      return EMPTY_ARRAY;
    }

    return modelConfigsAdapterSelectors.selectAll(data);
  }, [data]);

  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    isLoading,
    selectedModel: field.value,
    groupByType: true,
  });

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <FormControl className="nowheel nodrag" isDisabled={!options.length} isInvalid={!value}>
        <Combobox
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Flex>
  );
};

export default memo(ModelIdentifierFieldInputComponent);
