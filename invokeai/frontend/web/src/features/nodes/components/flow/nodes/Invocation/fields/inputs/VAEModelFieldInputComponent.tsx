import { Combobox, Flex, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldVaeModelValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { VAEModelFieldInputInstance, VAEModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<VAEModelFieldInputInstance, VAEModelFieldInputTemplate>;

const VAEModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useVAEModels();
  const _onChange = useCallback(
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
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: field.value,
    isLoading,
  });
  const required = props.fieldTemplate.required;

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <FormControl
        className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
        isDisabled={!options.length}
        isInvalid={!value && required}
      >
        <Combobox
          value={value}
          placeholder={required ? placeholder : `(Optional) ${placeholder}`}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Flex>
  );
};

export default memo(VAEModelFieldInputComponent);
