import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { fieldT2IAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  T2IAdapterModelInputFieldTemplate,
  T2IAdapterModelInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToT2IAdapterModelParam } from 'features/parameters/util/modelIdToT2IAdapterModelParam';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetT2IAdapterModelsQuery } from 'services/api/endpoints/models';

const T2IAdapterModelInputFieldComponent = (
  props: FieldComponentProps<
    T2IAdapterModelInputFieldValue,
    T2IAdapterModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const t2iAdapterModel = field.value;
  const dispatch = useAppDispatch();

  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery();

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      t2iAdapterModels?.entities[
        `${t2iAdapterModel?.base_model}/t2i_adapter/${t2iAdapterModel?.model_name}`
      ] ?? null,
    [
      t2iAdapterModel?.base_model,
      t2iAdapterModel?.model_name,
      t2iAdapterModels?.entities,
    ]
  );

  const data = useMemo(() => {
    if (!t2iAdapterModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(t2iAdapterModels.entities, (model, id) => {
      if (!model) {
        return;
      }

      data.push({
        value: id,
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [t2iAdapterModels]);

  const handleValueChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newT2IAdapterModel = modelIdToT2IAdapterModelParam(v);

      if (!newT2IAdapterModel) {
        return;
      }

      dispatch(
        fieldT2IAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newT2IAdapterModel,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <IAIMantineSelect
      className="nowheel nodrag"
      tooltip={selectedModel?.description}
      value={selectedModel?.id ?? null}
      placeholder="Pick one"
      error={!selectedModel}
      data={data}
      onChange={handleValueChanged}
      sx={{ width: '100%' }}
    />
  );
};

export default memo(T2IAdapterModelInputFieldComponent);
