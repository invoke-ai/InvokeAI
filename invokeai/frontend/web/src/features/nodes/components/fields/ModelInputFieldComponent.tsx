import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ModelInputFieldTemplate,
  ModelInputFieldValue,
} from 'features/nodes/types/types';

import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP as BASE_MODEL_NAME_MAP } from 'features/system/components/ModelSelect';
import { forEach, isString } from 'lodash-es';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';
import { FieldComponentProps } from './types';

const ModelInputFieldComponent = (
  props: FieldComponentProps<ModelInputFieldValue, ModelInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: mainModels } = useGetMainModelsQuery();

  const data = useMemo(() => {
    if (!mainModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(mainModels.entities, (model, id) => {
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
  }, [mainModels]);

  const selectedModel = useMemo(
    () => mainModels?.entities[field.value ?? mainModels.ids[0]],
    [mainModels?.entities, mainModels?.ids, field.value]
  );

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
    if (field.value && mainModels?.ids.includes(field.value)) {
      return;
    }

    const firstModel = mainModels?.ids[0];

    if (!isString(firstModel)) {
      return;
    }

    handleValueChanged(firstModel);
  }, [field.value, handleValueChanged, mainModels?.ids]);

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

export default memo(ModelInputFieldComponent);
