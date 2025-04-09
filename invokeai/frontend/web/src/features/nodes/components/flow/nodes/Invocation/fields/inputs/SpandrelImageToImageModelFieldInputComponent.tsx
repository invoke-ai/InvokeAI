import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldSpandrelImageToImageModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  SpandrelImageToImageModelFieldInputInstance,
  SpandrelImageToImageModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const SpandrelImageToImageModelFieldInputComponent = (
  props: FieldComponentProps<SpandrelImageToImageModelFieldInputInstance, SpandrelImageToImageModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const onChange = useCallback(
    (value: SpandrelImageToImageModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldSpandrelImageToImageModelValueChanged({
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

export default memo(SpandrelImageToImageModelFieldInputComponent);
