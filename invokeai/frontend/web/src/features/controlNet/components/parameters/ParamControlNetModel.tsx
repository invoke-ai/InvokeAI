import { useAppDispatch } from 'app/store/storeHooks';
import IAICustomSelect from 'common/components/IAICustomSelect';
import {
  CONTROLNET_MODELS,
  ControlNetModel,
} from 'features/controlNet/store/constants';
import { controlNetModelChanged } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetModelProps = {
  controlNetId: string;
  model: ControlNetModel;
};

const ParamControlNetModel = (props: ParamControlNetModelProps) => {
  const { controlNetId, model } = props;
  const dispatch = useAppDispatch();

  const handleModelChanged = useCallback(
    (val: string | null | undefined) => {
      // TODO: do not cast
      const model = val as ControlNetModel;
      dispatch(controlNetModelChanged({ controlNetId, model }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAICustomSelect
      tooltip={model}
      tooltipProps={{ placement: 'top', hasArrow: true }}
      items={CONTROLNET_MODELS}
      selectedItem={model}
      setSelectedItem={handleModelChanged}
      ellipsisPosition="start"
      withCheckIcon
    />
  );
};

export default memo(ParamControlNetModel);
