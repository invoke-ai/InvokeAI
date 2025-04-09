import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldT5EncoderValueChanged } from 'features/nodes/store/nodesSlice';
import type { T5EncoderModelFieldInputInstance, T5EncoderModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useT5EncoderModels } from 'services/api/hooks/modelsByType';
import type { T5EncoderBnbQuantizedLlmInt8bModelConfig, T5EncoderModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<T5EncoderModelFieldInputInstance, T5EncoderModelFieldInputTemplate>;

const T5EncoderModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useT5EncoderModels();
  const onChange = useCallback(
    (value: T5EncoderBnbQuantizedLlmInt8bModelConfig | T5EncoderModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldT5EncoderValueChanged({
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

export default memo(T5EncoderModelFieldInputComponent);
