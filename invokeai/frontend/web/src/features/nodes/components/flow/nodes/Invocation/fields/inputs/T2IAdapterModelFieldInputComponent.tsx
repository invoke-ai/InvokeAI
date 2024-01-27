import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldT2IAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { T2IAdapterModelConfigEntity } from 'services/api/endpoints/models';
import { useGetT2IAdapterModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

const T2IAdapterModelFieldInputComponent = (
  props: FieldComponentProps<T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery();

  const _onChange = useCallback(
    (value: T2IAdapterModelConfigEntity | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldT2IAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange } = useGroupedModelCombobox({
    modelEntities: t2iAdapterModels,
    onChange: _onChange,
    selectedModel: field.value ? { ...field.value, model_type: 't2i_adapter' } : undefined,
  });

  return (
    <Tooltip label={value?.description}>
      <FormControl className="nowheel nodrag" isInvalid={!value}>
        <Combobox value={value} placeholder="Pick one" options={options} onChange={onChange} />
      </FormControl>
    </Tooltip>
  );
};

export default memo(T2IAdapterModelFieldInputComponent);
