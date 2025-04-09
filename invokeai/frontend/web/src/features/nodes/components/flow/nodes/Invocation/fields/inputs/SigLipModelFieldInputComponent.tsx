import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldSigLipModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { SigLipModelFieldInputInstance, SigLipModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useSigLipModels } from 'services/api/hooks/modelsByType';
import type { SigLipModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const SigLipModelFieldInputComponent = (
  props: FieldComponentProps<SigLipModelFieldInputInstance, SigLipModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useSigLipModels();

  const onChange = useCallback(
    (value: SigLipModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldSigLipModelValueChanged({
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

export default memo(SigLipModelFieldInputComponent);
