import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { fieldIPAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  IPAdapterModelInputFieldTemplate,
  IPAdapterModelInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToIPAdapterModelParam } from 'features/parameters/util/modelIdToIPAdapterModelParams';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetIPAdapterModelsQuery } from 'services/api/endpoints/models';

const IPAdapterModelInputFieldComponent = (
  props: FieldComponentProps<
    IPAdapterModelInputFieldValue,
    IPAdapterModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const ipAdapterModel = field.value;
  const dispatch = useAppDispatch();

  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery();

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      ipAdapterModels?.entities[
        `${ipAdapterModel?.base_model}/ip_adapter/${ipAdapterModel?.model_name}`
      ] ?? null,
    [
      ipAdapterModel?.base_model,
      ipAdapterModel?.model_name,
      ipAdapterModels?.entities,
    ]
  );

  const data = useMemo(() => {
    if (!ipAdapterModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(ipAdapterModels.entities, (model, id) => {
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
  }, [ipAdapterModels]);

  const handleValueChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newIPAdapterModel = modelIdToIPAdapterModelParam(v);

      if (!newIPAdapterModel) {
        return;
      }

      dispatch(
        fieldIPAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newIPAdapterModel,
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

export default memo(IPAdapterModelInputFieldComponent);
