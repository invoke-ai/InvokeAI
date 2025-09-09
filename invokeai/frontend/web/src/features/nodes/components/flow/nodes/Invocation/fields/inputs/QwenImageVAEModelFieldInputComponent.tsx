import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldVaeModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { QwenImageVAEModelFieldInputInstance, QwenImageVAEModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useQwenImageVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<QwenImageVAEModelFieldInputInstance, QwenImageVAEModelFieldInputTemplate>;

const QwenImageVAEModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useQwenImageVAEModels();
  const onChange = useCallback(
    (value: VAEModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldVaeModelValueChanged({
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

export default memo(QwenImageVAEModelFieldInputComponent);