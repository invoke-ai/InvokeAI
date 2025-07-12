import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldMainModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  BriaControlNetModelFieldInputInstance,
  BriaControlNetModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useBriaControlNetModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<BriaControlNetModelFieldInputInstance, BriaControlNetModelFieldInputTemplate>;

const BriaControlNetModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useBriaControlNetModels();
  const onChange = useCallback(
    (value: ControlNetModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldMainModelValueChanged({
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

export default memo(BriaControlNetModelFieldInputComponent);
