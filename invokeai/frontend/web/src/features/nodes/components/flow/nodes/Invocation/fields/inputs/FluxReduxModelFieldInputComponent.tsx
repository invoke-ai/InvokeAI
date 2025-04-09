import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldFluxReduxModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { FluxReduxModelFieldInputInstance, FluxReduxModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useFluxReduxModels } from 'services/api/hooks/modelsByType';
import type { FLUXReduxModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const FluxReduxModelFieldInputComponent = (
  props: FieldComponentProps<FluxReduxModelFieldInputInstance, FluxReduxModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useFluxReduxModels();

  const onChange = useCallback(
    (value: FLUXReduxModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldFluxReduxModelValueChanged({
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

export default memo(FluxReduxModelFieldInputComponent);
