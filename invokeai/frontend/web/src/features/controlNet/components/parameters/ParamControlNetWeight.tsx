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

  const handleWeightReset = () => {
    dispatch(controlNetWeightChanged({ controlNetId, weight: 1 }));
  };

  if (mini) {
    return (
      <IAISlider
        label={'Weight'}
        sliderFormLabelProps={{ pb: 1 }}
        value={weight}
        onChange={handleWeightChanged}
        min={0}
        max={1}
        step={0.01}
      />
    );
  }

  return (
    <IAISlider
      label="Weight"
      value={weight}
      onChange={handleWeightChanged}
      withInput
      withReset
      handleReset={handleWeightReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.01}
    />
  );
};

export default memo(ParamControlNetWeight);
