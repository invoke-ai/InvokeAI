import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldControlLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  ControlLoRAModelFieldInputInstance,
  ControlLoRAModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useControlLoRAModel } from 'services/api/hooks/modelsByType';
import type { ControlLoRAModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ControlLoRAModelFieldInputInstance, ControlLoRAModelFieldInputTemplate>;

const ControlLoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlLoRAModel();

  const onChange = useCallback(
    (value: ControlLoRAModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldControlLoRAModelValueChanged({
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

export default memo(ControlLoRAModelFieldInputComponent);
