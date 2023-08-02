import { Box, Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToSDXLRefinerModelParam } from 'features/parameters/util/modelIdToSDXLRefinerModelParam';
import { refinerModelChanged } from 'features/sdxl/store/sdxlSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import SyncModelsButton from 'features/ui/components/tabs/ModelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selector = createSelector(
  stateSelector,
  (state) => ({ model: state.sdxl.refinerModel }),
  defaultSelectorOptions
);

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  const { model } = useAppSelector(selector);

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
      label="Refiner Model"
      placeholder="Loading..."
      disabled={true}
      data={[]}
    />
  ) : (
    <Flex w="100%" alignItems="center" gap={2}>
      <IAIMantineSearchableSelect
        tooltip={selectedModel?.description}
        label="Refiner Model"
        value={selectedModel?.id}
        placeholder={data.length > 0 ? 'Select a model' : 'No models available'}
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
