import { Combobox, Flex, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { SyncModelsIconButton } from 'features/modelManagerV2/components/SyncModels/SyncModelsIconButton';
import { fieldRefinerModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  SDXLRefinerModelFieldInputInstance,
  SDXLRefinerModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useRefinerModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<SDXLRefinerModelFieldInputInstance, SDXLRefinerModelFieldInputTemplate>;

const RefinerModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useRefinerModels();
  const _onChange = useCallback(
    (value: MainModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldRefinerModelValueChanged({
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
    isLoading,
    selectedModel: field.value,
  });

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <FormControl className="nowheel nodrag" isDisabled={!options.length} isInvalid={!value}>
        <Combobox
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
      <SyncModelsIconButton className="nodrag" />
    </Flex>
  );
};

export default memo(RefinerModelFieldInputComponent);
