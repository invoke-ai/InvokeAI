import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ControlNetModelInputFieldTemplate,
  ControlNetModelInputFieldValue,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP as BASE_MODEL_NAME_MAP } from 'features/system/components/ModelSelect';
import { forEach, isString } from 'lodash-es';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';
import { FieldComponentProps } from './types';

const ControlNetModelInputFieldComponent = (
  props: FieldComponentProps<
    ControlNetModelInputFieldValue,
    ControlNetModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: controlNetModels } = useGetControlNetModelsQuery();

  const selectedModel = useMemo(
    () => controlNetModels?.entities[field.value ?? controlNetModels.ids[0]],
    [controlNetModels?.entities, controlNetModels?.ids, field.value]
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
        label: model.name,
        group: BASE_MODEL_NAME_MAP[model.base_model],
      });
    });

    return data;
  }, [controlNetModels]);

  const handleValueChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: v,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  useEffect(() => {
    if (field.value && controlNetModels?.ids.includes(field.value)) {
      return;
    }

    const firstLora = controlNetModels?.ids[0];

    if (!isString(firstLora)) {
      return;
    }

    handleValueChanged(firstLora);
  }, [field.value, handleValueChanged, controlNetModels?.ids]);

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={
        selectedModel?.base_model &&
        BASE_MODEL_NAME_MAP[selectedModel?.base_model]
      }
      value={field.value}
      placeholder="Pick one"
      data={data}
      onChange={handleValueChanged}
    />
  );
};

export default memo(ControlNetModelInputFieldComponent);
