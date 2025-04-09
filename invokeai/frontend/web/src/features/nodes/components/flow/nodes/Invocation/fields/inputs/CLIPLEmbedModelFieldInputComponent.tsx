import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldCLIPLEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import type { CLIPLEmbedModelFieldInputInstance, CLIPLEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import { type CLIPLEmbedModelConfig, isCLIPLEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPLEmbedModelFieldInputInstance, CLIPLEmbedModelFieldInputTemplate>;

const CLIPLEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();

  const onChange = useCallback(
    (value: CLIPLEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPLEmbedValueChanged({
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
      modelConfigs={modelConfigs.filter((config) => isCLIPLEmbedModelConfig(config))}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
    />
  );
};

export default memo(CLIPLEmbedModelFieldInputComponent);
