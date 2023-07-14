import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSearchableSelect';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import {
  CONTROLNET_MODELS,
  ControlNetModelName,
} from 'features/controlNet/store/constants';
import { controlNetModelChanged } from 'features/controlNet/store/controlNetSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';

type ParamControlNetModelProps = {
  controlNetId: string;
  model: ControlNetModelName;
};

const selector = createSelector(configSelector, (config) => {
  const controlNetModels: IAISelectDataType[] = map(CONTROLNET_MODELS, (m) => ({
    label: m.label,
    value: m.type,
  })).filter(
    (d) =>
      !config.sd.disabledControlNetModels.includes(
        d.value as ControlNetModelName
      )
  );

  return controlNetModels;
});

const ParamControlNetModel = (props: ParamControlNetModelProps) => {
  const { controlNetId, model } = props;
  const controlNetModels = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();

  const handleModelChanged = useCallback(
    (val: string | null) => {
      // TODO: do not cast
      const model = val as ControlNetModelName;
      dispatch(controlNetModelChanged({ controlNetId, model }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      data={controlNetModels}
      value={model}
      onChange={handleModelChanged}
      disabled={!isReady}
      tooltip={model}
    />
  );
};

export default memo(ParamControlNetModel);
