import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldLLaVAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { LLaVAModelFieldInputInstance, LLaVAModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useLLaVAModels } from 'services/api/hooks/modelsByType';
import type { LlavaOnevisionConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<LLaVAModelFieldInputInstance, LLaVAModelFieldInputTemplate>;

const LLaVAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLLaVAModels();
  const onChange = useCallback(
    (value: LlavaOnevisionConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldLLaVAModelValueChanged({
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

export default memo(LLaVAModelFieldInputComponent);
