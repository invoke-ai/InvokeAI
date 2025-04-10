import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldControlNetModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { ControlNetModelFieldInputInstance, ControlNetModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useControlNetModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ControlNetModelFieldInputInstance, ControlNetModelFieldInputTemplate>;

const ControlNetModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlNetModels();

  const onChange = useCallback(
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

export default memo(ControlNetModelFieldInputComponent);
