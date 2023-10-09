import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setHrfScale } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';

export default function ParamHrfScale() {
  const { hrfScale, hrfEnabled } = useAppSelector((state: RootState) => ({
    hrfScale: state.generation.hrfScale,
    hrfEnabled: state.generation.hrfEnabled,
  }));
  const dispatch = useAppDispatch();

  const handleHrfScaleReset = useCallback(() => {
    dispatch(setHrfScale(1));
  }, [dispatch]);

  const handleHrfScaleChange = useCallback(
    (v: number) => {
      dispatch(setHrfScale(v));
    },
    [dispatch]
  );

  return (
    <IAISlider
      label="High Resolution Fix Scale"
      aria-label="High Fix Resolution Scale"
      min={1}
      max={5}
      step={1}
      value={hrfScale}
      onChange={handleHrfScaleChange}
      withSliderMarks
      withInput
      withReset
      handleReset={handleHrfScaleReset}
      isDisabled={!hrfEnabled}
    />
  );
}
