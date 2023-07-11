import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import { controlNetModelChanged } from 'features/controlNet/store/controlNetSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';

type ParamControlNetModelProps = {
  controlNetId: string;
  model: string;
};

const ParamControlNetModel = (props: ParamControlNetModelProps) => {
  const { controlNetId, model } = props;
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();

  const currentMainModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const { data: controlNetModels } = useGetControlNetModelsQuery();

  const handleModelChanged = useCallback(
    (val: string | null) => {
      if (!val) return;
      dispatch(controlNetModelChanged({ controlNetId, model: val }));
    },
    [controlNetId, dispatch]
  );

  const data = useMemo(() => {
    if (!controlNetModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(controlNetModels.entities, (model, id) => {
      if (!model) {
        return;
      }

      const disabled = currentMainModel?.base_model !== model.base_model;

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
  }, [controlNetModels, currentMainModel?.base_model]);

  return (
    <IAIMantineSearchableSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      data={data}
      value={model}
      onChange={handleModelChanged}
      disabled={!isReady}
      tooltip={model}
    />
  );
};

export default memo(ParamControlNetModel);
