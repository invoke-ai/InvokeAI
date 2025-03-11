import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldLLaVAModelValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
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
  const _onChange = useCallback(
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

  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: field.value,
    isLoading,
  });

  return (
    <FormControl className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`} isInvalid={!value} isDisabled={!options.length}>
      <Combobox
        value={value}
        placeholder={placeholder}
        noOptionsMessage={noOptionsMessage}
        options={options}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(LLaVAModelFieldInputComponent);
