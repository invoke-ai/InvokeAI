import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';

import { Box, Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { modelSelected } from 'features/parameters/store/actions';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToMainModelParam } from 'features/parameters/util/modelIdToMainModelParam';
import SyncModelsButton from 'features/ui/components/tabs/ModelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { forEach } from 'lodash-es';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import {
  useGetMainModelsQuery,
  useGetOnnxModelsQuery,
} from 'services/api/endpoints/models';
import { useFeatureStatus } from '../../../../system/hooks/useFeatureStatus';

const selector = createSelector(
  stateSelector,
  (state) => ({ model: state.generation.model }),
  defaultSelectorOptions
);

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { model } = useAppSelector(selector);

  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;
  const { data: mainModels, isLoading } = useGetMainModelsQuery(
    NON_REFINER_BASE_MODELS
  );
  const { data: onnxModels, isLoading: onnxLoading } = useGetOnnxModelsQuery(
    NON_REFINER_BASE_MODELS
  );

  const activeTabName = useAppSelector(activeTabNameSelector);

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
    forEach(onnxModels?.entities, (model, id) => {
      if (
        !model ||
        activeTabName === 'unifiedCanvas' ||
        activeTabName === 'img2img'
      ) {
        return;
      }

      data.push({
        value: id,
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [mainModels, onnxModels, activeTabName]);

  // grab the full model entity from the RTK Query cache
  // TODO: maybe we should just store the full model entity in state?
  const selectedModel = useMemo(
    () =>
      (mainModels?.entities[`${model?.base_model}/main/${model?.model_name}`] ||
        onnxModels?.entities[
          `${model?.base_model}/onnx/${model?.model_name}`
        ]) ??
      null,
    [mainModels?.entities, model, onnxModels?.entities]
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

      dispatch(modelSelected(newModel));
    },
    [dispatch]
  );

  return isLoading || onnxLoading ? (
    <IAIMantineSearchableSelect
      label={t('modelManager.model')}
      placeholder="Loading..."
      disabled={true}
      data={[]}
    />
  ) : (
    <Flex w="100%" alignItems="center" gap={3}>
      <IAIMantineSearchableSelect
        tooltip={selectedModel?.description}
        label={t('modelManager.model')}
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

export default memo(ParamMainModelSelect);
