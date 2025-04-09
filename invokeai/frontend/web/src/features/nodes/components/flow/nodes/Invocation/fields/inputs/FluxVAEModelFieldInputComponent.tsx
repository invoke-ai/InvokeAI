import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldFluxVAEModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { FluxVAEModelFieldInputInstance, FluxVAEModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useFluxVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<FluxVAEModelFieldInputInstance, FluxVAEModelFieldInputTemplate>;

const FluxVAEModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useFluxVAEModels();
  const onChange = useCallback(
    (value: VAEModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldFluxVAEModelValueChanged({
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

export default memo(FluxVAEModelFieldInputComponent);
