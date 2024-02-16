import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldIPAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { IPAdapterModelFieldInputInstance, IPAdapterModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { IPAdapterConfig } from 'services/api/endpoints/models';
import { useGetIPAdapterModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

const IPAdapterModelFieldInputComponent = (
  props: FieldComponentProps<IPAdapterModelFieldInputInstance, IPAdapterModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery();

  const _onChange = useCallback(
    (value: IPAdapterConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldIPAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange } = useGroupedModelCombobox({
    modelEntities: ipAdapterModels,
    onChange: _onChange,
    selectedModel: field.value ? { ...field.value, model_type: 'ip_adapter' } : undefined,
  });

  return (
    <Tooltip label={value?.description}>
      <FormControl className="nowheel nodrag" isInvalid={!value}>
        <Combobox value={value} placeholder="Pick one" options={options} onChange={onChange} />
      </FormControl>
    </Tooltip>
  );
};

export default memo(IPAdapterModelFieldInputComponent);
