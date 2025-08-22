import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldRunwayModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { RunwayModelFieldInputInstance, RunwayModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useRunwayModels } from 'services/api/hooks/modelsByType';
import type { VideoApiModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const RunwayModelFieldInputComponent = (
  props: FieldComponentProps<RunwayModelFieldInputInstance, RunwayModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useRunwayModels();

  const onChange = useCallback(
    (value: VideoApiModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldRunwayModelValueChanged({
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

export default memo(RunwayModelFieldInputComponent);
