import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { fieldControlNetModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ControlNetModelInputFieldTemplate,
  ControlNetModelInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToControlNetModelParam } from 'features/parameters/util/modelIdToControlNetModelParam';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';

const ControlNetModelInputFieldComponent = (
  props: FieldComponentProps<
    ControlNetModelInputFieldValue,
    ControlNetModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const controlNetModel = field.value;
  const dispatch = useAppDispatch();

  const { data: controlNetModels } = useGetControlNetModelsQuery();

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      controlNetModels?.entities[
        `${controlNetModel?.base_model}/controlnet/${controlNetModel?.model_name}`
      ] ?? null,
    [
      controlNetModel?.base_model,
      controlNetModel?.model_name,
      controlNetModels?.entities,
    ]
  );

  const data = useMemo(() => {
    if (!controlNetModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(controlNetModels.entities, (model, id) => {
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
  }, [controlNetModels]);

  const handleValueChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newControlNetModel = modelIdToControlNetModelParam(v);

      if (!newControlNetModel) {
        return;
      }

      dispatch(
        fieldControlNetModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newControlNetModel,
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

export default memo(ControlNetModelInputFieldComponent);
