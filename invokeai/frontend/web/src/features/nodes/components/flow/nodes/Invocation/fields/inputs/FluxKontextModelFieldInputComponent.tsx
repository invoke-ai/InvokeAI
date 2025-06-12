import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldFluxKontextModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  FluxKontextModelFieldInputInstance,
  FluxKontextModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useFluxKontextModels } from 'services/api/hooks/modelsByType';
import type { ApiModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const FluxKontextModelFieldInputComponent = (
  props: FieldComponentProps<FluxKontextModelFieldInputInstance, FluxKontextModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useFluxKontextModels();

  const onChange = useCallback(
    (value: ApiModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldFluxKontextModelValueChanged({
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

export default memo(FluxKontextModelFieldInputComponent);
