import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setHrfScale } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';

export default function ParamHrfScale() {
  const hrfScale = useAppSelector(
    (state: RootState) => state.generation.hrfScale
  );
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
      label="High Resolution Scale"
      aria-label="High Resolution Scale"
      min={1}
      max={5}
      step={0.1}
      value={hrfScale}
      onChange={handleHrfScaleChange}
      withSliderMarks
      withInput
      withReset
      handleReset={handleHrfScaleReset}
    />
  );
}
