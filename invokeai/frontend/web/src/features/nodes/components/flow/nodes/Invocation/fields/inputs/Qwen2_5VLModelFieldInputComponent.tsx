import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldMainModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { Qwen2_5VLModelFieldInputInstance, Qwen2_5VLModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useMainModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<Qwen2_5VLModelFieldInputInstance, Qwen2_5VLModelFieldInputTemplate>;

const Qwen2_5VLModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  // For now, using main models as Qwen2.5-VL is a main model that acts as text encoder
  // In the future, we might want to create a specific hook for Qwen2.5-VL models
  const [modelConfigs, { isLoading }] = useMainModels();
  const onChange = useCallback(
    (value: MainModelConfig | null) => {
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

export default memo(Qwen2_5VLModelFieldInputComponent);