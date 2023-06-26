import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ModelInputFieldTemplate,
  ModelInputFieldValue,
} from 'features/nodes/types/types';

import { memo, useCallback, useEffect, useMemo } from 'react';
import { FieldComponentProps } from './types';
import { forEach, isString } from 'lodash-es';
import { MODEL_TYPE_MAP as BASE_MODEL_NAME_MAP } from 'features/system/components/ModelSelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useTranslation } from 'react-i18next';
import { useListModelsQuery } from 'services/api/endpoints/models';

const ModelInputFieldComponent = (
  props: FieldComponentProps<ModelInputFieldValue, ModelInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: pipelineModels } = useListModelsQuery({
    model_type: 'pipeline',
  });

  const data = useMemo(() => {
    if (!pipelineModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(pipelineModels.entities, (model, id) => {
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
  }, [pipelineModels]);

  const selectedModel = useMemo(
    () => pipelineModels?.entities[field.value ?? pipelineModels.ids[0]],
    [pipelineModels?.entities, pipelineModels?.ids, field.value]
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
    if (field.value && pipelineModels?.ids.includes(field.value)) {
      return;
    }

    const firstModel = pipelineModels?.ids[0];

    if (!isString(firstModel)) {
      return;
    }

    handleValueChanged(firstModel);
  }, [field.value, handleValueChanged, pipelineModels?.ids]);

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
