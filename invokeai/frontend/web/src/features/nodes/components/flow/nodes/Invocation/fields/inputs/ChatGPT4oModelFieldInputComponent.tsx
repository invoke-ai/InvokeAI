import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldChatGPT4oModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { ChatGPT4oModelFieldInputInstance, ChatGPT4oModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useChatGPT4oModels } from 'services/api/hooks/modelsByType';
import type { ApiModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const ChatGPT4oModelFieldInputComponent = (
  props: FieldComponentProps<ChatGPT4oModelFieldInputInstance, ChatGPT4oModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useChatGPT4oModels();

  const onChange = useCallback(
    (value: ApiModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldChatGPT4oModelValueChanged({
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

export default memo(ChatGPT4oModelFieldInputComponent);
