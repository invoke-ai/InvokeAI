import { useAppDispatch } from 'app/store/storeHooks';
import IAICustomSelect, {
  IAICustomSelectOption,
} from 'common/components/IAICustomSelect';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import {
  CONTROLNET_MODELS,
  ControlNetModelName,
} from 'features/controlNet/store/constants';
import { controlNetModelChanged } from 'features/controlNet/store/controlNetSlice';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';

type ParamControlNetModelProps = {
  controlNetId: string;
  model: ControlNetModelName;
};

const DATA: IAICustomSelectOption[] = map(CONTROLNET_MODELS, (m) => ({
  value: m.type,
  label: m.label,
  tooltip: m.type,
}));

const ParamControlNetModel = (props: ParamControlNetModelProps) => {
  const { controlNetId, model } = props;
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();

  const handleModelChanged = useCallback(
    (val: string | null | undefined) => {
      // TODO: do not cast
      const model = val as ControlNetModelName;
      dispatch(controlNetModelChanged({ controlNetId, model }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAICustomSelect
      tooltip={model}
      tooltipProps={{ placement: 'top', hasArrow: true }}
      data={DATA}
      value={model}
      onChange={handleModelChanged}
      isDisabled={!isReady}
      ellipsisPosition="start"
      withCheckIcon
    />
  );
};

export default memo(ParamControlNetModel);
