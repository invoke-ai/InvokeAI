import { Box, Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import SyncModelsButton from 'features/modelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToSDXLRefinerModelParam } from 'features/parameters/util/modelIdToSDXLRefinerModelParam';
import { refinerModelChanged } from 'features/sdxl/store/sdxlSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(stateSelector, (state) => ({
  model: state.sdxl.refinerModel,
}));

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  const { model } = useAppSelector(selector);
  const { t } = useTranslation();

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
        `${model?.base_model}/main/${model?.model_name}`
      ] ?? null,
    [refinerModels?.entities, model]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newModel = modelIdToSDXLRefinerModelParam(v);

      if (!newModel) {
        return;
      }

      dispatch(refinerModelChanged(newModel));
    },
    [dispatch]
  );

  return isLoading ? (
    <IAIMantineSearchableSelect
      label={t('sdxl.refinermodel')}
      placeholder={t('sdxl.loading')}
      disabled={true}
      data={[]}
    />
  ) : (
    <Flex w="100%" alignItems="center" gap={2}>
      <IAIMantineSearchableSelect
        tooltip={selectedModel?.description}
        label={t('sdxl.refinermodel')}
        value={selectedModel?.id}
        placeholder={
          data.length > 0 ? t('sdxl.selectAModel') : t('sdxl.noModelsAvailable')
        }
        data={data}
        error={data.length === 0}
        disabled={data.length === 0}
        onChange={handleChangeModel}
        w="100%"
      />
      {isSyncModelEnabled && (
        <Box mt={7}>
          <SyncModelsButton iconMode />
        </Box>
      )}
    </Flex>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
