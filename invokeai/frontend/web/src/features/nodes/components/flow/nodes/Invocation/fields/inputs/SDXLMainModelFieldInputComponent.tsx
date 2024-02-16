import { Combobox, Flex, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { SyncModelsIconButton } from 'features/modelManager/components/SyncModels/SyncModelsIconButton';
import { fieldMainModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { SDXLMainModelFieldInputInstance, SDXLMainModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { SDXL_MAIN_MODELS } from 'services/api/constants';
import type { MainModelConfig } from 'services/api/endpoints/models';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<SDXLMainModelFieldInputInstance, SDXLMainModelFieldInputTemplate>;

const SDXLMainModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetMainModelsQuery(SDXL_MAIN_MODELS);
  const _onChange = useCallback(
    (value: MainModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldMainModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelEntities: data,
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

export default memo(SDXLMainModelFieldInputComponent);
