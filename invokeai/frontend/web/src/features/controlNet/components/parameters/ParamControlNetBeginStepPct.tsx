import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import {
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetBeginStepPctProps = {
  controlNetId: string;
  beginStepPct: number;
};

const ParamControlNetBeginStepPct = (
  props: ParamControlNetBeginStepPctProps
) => {
  const { controlNetId, beginStepPct } = props;
  const dispatch = useAppDispatch();

  const handleBeginStepPctChanged = useCallback(
    (beginStepPct: number) => {
      dispatch(controlNetBeginStepPctChanged({ controlNetId, beginStepPct }));
    },
    [controlNetId, dispatch]
  );

  const handleBeginStepPctReset = useCallback(() => {
    dispatch(controlNetBeginStepPctChanged({ controlNetId, beginStepPct: 0 }));
  }, [controlNetId, dispatch]);

  const handleEndStepPctChanged = useCallback(
    (endStepPct: number) => {
      dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct }));
    },
    [controlNetId, dispatch]
  );

  const handleEndStepPctReset = useCallback(() => {
    dispatch(controlNetEndStepPctChanged({ controlNetId, endStepPct: 0 }));
  }, [controlNetId, dispatch]);

  return (
    <IAISlider
      label="Begin Step %"
      value={beginStepPct}
      onChange={handleBeginStepPctChanged}
      withInput
      withReset
      handleReset={handleBeginStepPctReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.01}
    />
  );
};

export default memo(ParamControlNetBeginStepPct);
