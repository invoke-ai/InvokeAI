import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
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
  const onChange = useCallback(
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

  return (
    <ModelFieldCombobox
      value={field.value}
      modelConfigs={modelConfigs}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
      groupByType
    />
  );
};

export default memo(ModelIdentifierFieldInputComponent);
