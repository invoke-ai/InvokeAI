import { Flex, Text } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { fieldMainModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  MainModelInputFieldTemplate,
  MainModelInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToMainModelParam } from 'features/parameters/util/modelIdToMainModelParam';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import SyncModelsButton from 'features/ui/components/tabs/ModelManager/subpanels/ModelManagerSettingsPanel/SyncModelsButton';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { NON_SDXL_MAIN_MODELS } from 'services/api/constants';
import {
  useGetMainModelsQuery,
  useGetOnnxModelsQuery,
} from 'services/api/endpoints/models';

const MainModelInputFieldComponent = (
  props: FieldComponentProps<
    MainModelInputFieldValue,
    MainModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  const { data: onnxModels, isLoading: isLoadingOnnxModels } =
    useGetOnnxModelsQuery(NON_SDXL_MAIN_MODELS);
  const { data: mainModels, isLoading: isLoadingMainModels } =
    useGetMainModelsQuery(NON_SDXL_MAIN_MODELS);

  const isLoadingModels = useMemo(
    () => isLoadingOnnxModels || isLoadingMainModels,
    [isLoadingOnnxModels, isLoadingMainModels]
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

    if (onnxModels) {
      forEach(onnxModels.entities, (model, id) => {
        if (!model) {
          return;
        }

        data.push({
          value: id,
          label: model.model_name,
          group: MODEL_TYPE_MAP[model.base_model],
        });
      });
    }
    return data;
  }, [mainModels, onnxModels]);

  // grab the full model entity from the RTK Query cache
  // TODO: maybe we should just store the full model entity in state?
  const selectedModel = useMemo(
    () =>
      (mainModels?.entities[
        `${field.value?.base_model}/main/${field.value?.model_name}`
      ] ||
        onnxModels?.entities[
          `${field.value?.base_model}/onnx/${field.value?.model_name}`
        ]) ??
      null,
    [
      field.value?.base_model,
      field.value?.model_name,
      mainModels?.entities,
      onnxModels?.entities,
    ]
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
        fieldMainModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newModel,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <Flex sx={{ w: 'full', alignItems: 'center', gap: 2 }}>
      {isLoadingModels ? (
        <Text variant="subtext">Loading...</Text>
      ) : (
        <IAIMantineSearchableSelect
          className="nowheel nodrag"
          tooltip={selectedModel?.description}
          value={selectedModel?.id}
          placeholder={
            data.length > 0 ? 'Select a model' : 'No models available'
          }
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
      )}
      {isSyncModelEnabled && <SyncModelsButton className="nodrag" iconMode />}
    </Flex>
  );
};

export default memo(MainModelInputFieldComponent);
