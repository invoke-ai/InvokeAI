import { Combobox, Flex, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { SyncModelsIconButton } from 'features/modelManager/components/SyncModels/SyncModelsIconButton';
import { fieldVaeModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { VAEModelFieldInputInstance, VAEModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { VAEConfig } from 'services/api/endpoints/models';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<VAEModelFieldInputInstance, VAEModelFieldInputTemplate>;

const VAEModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetVaeModelsQuery();
  const _onChange = useCallback(
    (value: VAEConfig | null) => {
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
    modelEntities: data,
    onChange: _onChange,
    selectedModel: field.value ? { ...field.value, model_type: 'vae' } : null,
    isLoading,
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

export default memo(VAEModelFieldInputComponent);
