import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldVeo3ModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { Veo3ModelFieldInputInstance, Veo3ModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useVeo3Models } from 'services/api/hooks/modelsByType';
import type { VideoApiModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const Veo3ModelFieldInputComponent = (
  props: FieldComponentProps<Veo3ModelFieldInputInstance, Veo3ModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useVeo3Models();

  const onChange = useCallback(
    (value: VideoApiModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldVeo3ModelValueChanged({
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

export default memo(Veo3ModelFieldInputComponent);
