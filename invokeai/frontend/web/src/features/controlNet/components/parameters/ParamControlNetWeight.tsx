import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import {
  ControlNetConfig,
  controlNetWeightChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetWeightProps = {
  controlNet: ControlNetConfig;
};

const ParamControlNetWeight = (props: ParamControlNetWeightProps) => {
  const { weight, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();
  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(controlNetWeightChanged({ controlNetId, weight }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAISlider
      isDisabled={!isEnabled}
      label="Weight"
      value={weight}
      onChange={handleWeightChanged}
      min={0}
      max={2}
      step={0.01}
      withSliderMarks
      sliderMarks={[0, 1, 2]}
    />
  );
};

export default memo(ParamControlNetWeight);
