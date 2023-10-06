import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setHrf } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';

export default function ParamHrf() {
  const hrfScale = useAppSelector(
    (state: RootState) => state.generation.hrfScale
  );
  const dispatch = useAppDispatch();

  const handleHrfSkipReset = useCallback(() => {
    dispatch(setHrf(0));
  }, [dispatch]);

  const handleHrfChange = useCallback(
    (v: number) => {
      dispatch(setHrf(v));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="hrf" placement="top">
      <IAISlider
        label="High Resolution Scale"
        aria-label="High Resolution Scale"
        min={1}
        max={20}
        step={0.1}
        value={hrfScale}
        onChange={handleHrfChange}
        withSliderMarks
        withInput
        withReset
        handleReset={handleHrfSkipReset}
      />
    </IAIInformationalPopover>
  );
}
