import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldCLIPEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import type { CLIPEmbedModelFieldInputInstance, CLIPEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import type { CLIPEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPEmbedModelFieldInputInstance, CLIPEmbedModelFieldInputTemplate>;

const CLIPEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();
  const onChange = useCallback(
    (value: CLIPEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPEmbedValueChanged({
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

export default memo(CLIPEmbedModelFieldInputComponent);
