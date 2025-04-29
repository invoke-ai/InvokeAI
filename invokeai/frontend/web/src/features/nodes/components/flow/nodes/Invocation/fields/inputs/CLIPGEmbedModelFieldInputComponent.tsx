import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldCLIPGEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import type { CLIPGEmbedModelFieldInputInstance, CLIPGEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import { type CLIPGEmbedModelConfig, isCLIPGEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPGEmbedModelFieldInputInstance, CLIPGEmbedModelFieldInputTemplate>;

const CLIPGEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();

  const onChange = useCallback(
    (value: CLIPGEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPGEmbedValueChanged({
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
      modelConfigs={modelConfigs.filter((config) => isCLIPGEmbedModelConfig(config))}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
    />
  );
};

export default memo(CLIPGEmbedModelFieldInputComponent);
