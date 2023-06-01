import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { controlNetEndStepPctChanged } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetEndStepPctProps = {
  controlNetId: string;
  endStepPct: number;
};

const ParamControlNetEndStepPct = (props: ParamControlNetEndStepPctProps) => {
  const { controlNetId, endStepPct } = props;
  const dispatch = useAppDispatch();

  const handleEndStepPctChanged = useCallback(
    (endStepPct: number) => {
      dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct }));
    },
    [controlNetId, dispatch]
  );

  const handleEndStepPctReset = () => {
    dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct: 0 }));
  };

  return (
    <IAISlider
      label="End Step %"
      value={endStepPct}
      onChange={handleEndStepPctChanged}
      withInput
      withReset
      handleReset={handleEndStepPctReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.01}
    />
  );
};

export default memo(ParamControlNetEndStepPct);
