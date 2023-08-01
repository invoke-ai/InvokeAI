import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  MainModelInputFieldValue,
  ModelInputFieldTemplate,
} from 'features/nodes/types/types';

import { Box, Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToMainModelParam } from 'features/parameters/util/modelIdToMainModelParam';
import SyncModelsButton from 'features/ui/components/tabs/ModelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';
import { FieldComponentProps } from './types';
import { useFeatureStatus } from '../../../system/hooks/useFeatureStatus';

const ModelInputFieldComponent = (
  props: FieldComponentProps<MainModelInputFieldValue, ModelInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  const { data: mainModels, isLoading } = useGetMainModelsQuery(
    NON_REFINER_BASE_MODELS
  );

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
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [mainModels]);

  // grab the full model entity from the RTK Query cache
  // TODO: maybe we should just store the full model entity in state?
  const selectedModel = useMemo(
    () =>
      mainModels?.entities[
        `${field.value?.base_model}/main/${field.value?.model_name}`
      ] ?? null,
    [field.value?.base_model, field.value?.model_name, mainModels?.entities]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newModel = modelIdToMainModelParam(v);

      if (!newModel) {
        return;
      }

      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: newModel,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return isLoading ? (
    <IAIMantineSearchableSelect
      label={t('modelManager.model')}
      placeholder="Loading..."
      disabled={true}
      data={[]}
    />
  ) : (
    <Flex w="100%" alignItems="center" gap={2}>
      <IAIMantineSearchableSelect
        tooltip={selectedModel?.description}
        label={
          selectedModel?.base_model && MODEL_TYPE_MAP[selectedModel?.base_model]
        }
        value={selectedModel?.id}
        placeholder={data.length > 0 ? 'Select a model' : 'No models available'}
        data={data}
        error={data.length === 0}
        disabled={data.length === 0}
        onChange={handleChangeModel}
      />
      {isSyncModelEnabled && (
        <Box mt={7}>
          <SyncModelsButton iconMode />
        </Box>
      )}
    </Flex>
  );
};

export default memo(ModelInputFieldComponent);
