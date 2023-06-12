import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICustomSelect, {
  IAICustomSelectOption,
} from 'common/components/IAICustomSelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import IAISelect from 'common/components/IAISelect';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import {
  CONTROLNET_MODELS,
  ControlNetModelName,
} from 'features/controlNet/store/constants';
import { controlNetModelChanged } from 'features/controlNet/store/controlNetSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { map } from 'lodash-es';
import { ChangeEvent, memo, useCallback } from 'react';

type ParamControlNetModelProps = {
  controlNetId: string;
  model: ControlNetModelName;
};

const selector = createSelector(configSelector, (config) => {
  return map(CONTROLNET_MODELS, (m) => ({
    label: m.label,
    value: m.type,
  })).filter((d) => !config.sd.disabledControlNetModels.includes(d.value));
});

const DATA = map(CONTROLNET_MODELS, (m) => ({
  value: m.type,
  label: m.label,
  tooltip: m.type,
}));

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
    <IAIMantineSelect
      data={controlNetModels}
      value={model}
      onChange={handleModelChanged}
      disabled={!isReady}
      tooltip={model}
    />
  );
  // return (
  //   <IAICustomSelect
  //     tooltip={model}
  //     tooltipProps={{ placement: 'top', hasArrow: true }}
  //     data={DATA}
  //     value={model}
  //     onChange={handleModelChanged}
  //     isDisabled={!isReady}
  //     ellipsisPosition="start"
  //     withCheckIcon
  //   />
  // );
};

export default memo(ParamControlNetModel);
