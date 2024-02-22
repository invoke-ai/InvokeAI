import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldT2IAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate } from 'features/nodes/types/field';
import { pick } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useGetT2IAdapterModelsQuery } from 'services/api/endpoints/models';
import type { T2IAdapterModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const T2IAdapterModelFieldInputComponent = (
  props: FieldComponentProps<T2IAdapterModelFieldInputInstance, T2IAdapterModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery();

  const _onChange = useCallback(
    (value: T2IAdapterModelConfig | null) => {
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
    selectedModel: field.value ? pick(field.value, ['key', 'base']) : undefined,
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
