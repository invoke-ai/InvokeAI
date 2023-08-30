import { Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { fieldRefinerModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  FieldComponentProps,
  SDXLRefinerModelInputFieldTemplate,
  SDXLRefinerModelInputFieldValue,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToMainModelParam } from 'features/parameters/util/modelIdToMainModelParam';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import SyncModelsButton from 'features/ui/components/tabs/ModelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const RefinerModelInputFieldComponent = (
  props: FieldComponentProps<
    SDXLRefinerModelInputFieldValue,
    SDXLRefinerModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;
  const { data: refinerModels, isLoading } =
    useGetMainModelsQuery(REFINER_BASE_MODELS);

  const data = useMemo(() => {
    if (!refinerModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(refinerModels.entities, (model, id) => {
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
  }, [refinerModels]);

  // grab the full model entity from the RTK Query cache
  // TODO: maybe we should just store the full model entity in state?
  const selectedModel = useMemo(
    () =>
      refinerModels?.entities[
        `${field.value?.base_model}/main/${field.value?.model_name}`
      ] ?? null,
    [field.value?.base_model, field.value?.model_name, refinerModels?.entities]
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
        fieldRefinerModelValueChanged({
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
        className="nowheel nodrag"
        tooltip={selectedModel?.description}
        value={selectedModel?.id}
        placeholder={data.length > 0 ? 'Select a model' : 'No models available'}
        data={data}
        error={!selectedModel}
        disabled={data.length === 0}
        onChange={handleChangeModel}
        sx={{
          width: '100%',
          '.mantine-Select-dropdown': {
            width: '16rem !important',
          },
        }}
      />
      {isSyncModelEnabled && <SyncModelsButton className="nodrag" iconMode />}
    </Flex>
  );
};

export default memo(RefinerModelInputFieldComponent);
