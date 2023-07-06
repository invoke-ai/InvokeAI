import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  VaeModelInputFieldTemplate,
  VaeModelInputFieldValue,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP as BASE_MODEL_NAME_MAP } from 'features/system/components/ModelSelect';
import { forEach } from 'lodash-es';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';
import { FieldComponentProps } from './types';

const VaeModelInputFieldComponent = (
  props: FieldComponentProps<
    VaeModelInputFieldValue,
    VaeModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: vaeModels } = useGetVaeModelsQuery();

  const selectedModel = useMemo(
    () => vaeModels?.entities[field.value ?? vaeModels.ids[0]],
    [vaeModels?.entities, vaeModels?.ids, field.value]
  );

  const data = useMemo(() => {
    if (!vaeModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(vaeModels.entities, (model, id) => {
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
  }, [vaeModels]);

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
    if (field.value && vaeModels?.ids.includes(field.value)) {
      return;
    }
    handleValueChanged('auto');
  }, [field.value, handleValueChanged, vaeModels?.ids]);

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

export default memo(VaeModelInputFieldComponent);
