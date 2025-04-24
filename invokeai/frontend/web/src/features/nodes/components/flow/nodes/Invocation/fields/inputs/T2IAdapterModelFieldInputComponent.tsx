import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldT2IAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useT2IAdapterModels } from 'services/api/hooks/modelsByType';
import type { T2IAdapterModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const T2IAdapterModelFieldInputComponent = (
  props: FieldComponentProps<T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useT2IAdapterModels();

  const onChange = useCallback(
    (value: T2IAdapterModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldT2IAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <ModelFieldCombobox
      value={field.value}
      modelConfigs={modelConfigs}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
    />
  );
};

export default memo(T2IAdapterModelFieldInputComponent);
