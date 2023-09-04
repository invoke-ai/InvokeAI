import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import {
  ControlNetConfig,
  controlNetModelChanged,
} from 'features/controlNet/store/controlNetSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToControlNetModelParam } from 'features/parameters/util/modelIdToControlNetModelParam';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';

type ParamControlNetModelProps = {
  controlNet: ControlNetConfig;
};

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { model } = generation;
    return { mainModel: model };
  },
  defaultSelectorOptions
);

const ParamControlNetModel = (props: ParamControlNetModelProps) => {
  const { controlNetId, model: controlNetModel, isEnabled } = props.controlNet;
  const dispatch = useAppDispatch();
  const isBusy = useAppSelector(selectIsBusy);

  const { mainModel } = useAppSelector(selector);

  const { data: controlNetModels } = useGetControlNetModelsQuery();

  const data = useMemo(() => {
    if (!controlNetModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(controlNetModels.entities, (model, id) => {
      if (!model) {
        return;
      }

      const disabled = model?.base_model !== mainModel?.base_model;

      data.push({
        value: id,
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
        disabled,
        tooltip: disabled
          ? `Incompatible base model: ${model.base_model}`
          : undefined,
      });
    });

    return data;
  }, [controlNetModels, mainModel?.base_model]);

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      controlNetModels?.entities[
        `${controlNetModel?.base_model}/controlnet/${controlNetModel?.model_name}`
      ] ?? null,
    [
      controlNetModel?.base_model,
      controlNetModel?.model_name,
      controlNetModels?.entities,
    ]
  );

  const handleModelChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newControlNetModel = modelIdToControlNetModelParam(v);

      if (!newControlNetModel) {
        return;
      }

      dispatch(
        controlNetModelChanged({ controlNetId, model: newControlNetModel })
      );
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      data={data}
      error={
        !selectedModel || mainModel?.base_model !== selectedModel.base_model
      }
      placeholder="Select a model"
      value={selectedModel?.id ?? null}
      onChange={handleModelChanged}
      disabled={isBusy || !isEnabled}
      tooltip={selectedModel?.description}
    />
  );
};

export default memo(ParamControlNetModel);
