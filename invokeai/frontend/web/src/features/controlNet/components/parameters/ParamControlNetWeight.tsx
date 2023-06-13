import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { controlNetWeightChanged } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetWeightProps = {
  controlNetId: string;
  weight: number;
  mini?: boolean;
};

const ParamControlNetWeight = (props: ParamControlNetWeightProps) => {
  const { controlNetId, weight, mini = false } = props;
  const dispatch = useAppDispatch();

  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(controlNetWeightChanged({ controlNetId, weight }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAISlider
      label={'Weight'}
      sliderFormLabelProps={{ pb: 2 }}
      value={weight}
      onChange={handleWeightChanged}
      min={-1}
      max={1}
      step={0.01}
      withSliderMarks={!mini}
      sliderMarks={[-1, 0, 1]}
    />
  );
};

export default memo(ParamControlNetWeight);
