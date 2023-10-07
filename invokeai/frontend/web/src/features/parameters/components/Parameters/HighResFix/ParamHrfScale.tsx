import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import {
  setHrfScale,
  setHrfToggle,
} from 'features/parameters/store/generationSlice';
import { useCallback, ChangeEvent } from 'react';

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

  const hrfToggled = useAppSelector(
    (state: RootState) => state.generation.hrfToggled
  );

  const handleHrfToggle = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHrfToggle(e.target.checked));

  return (
    <IAIInformationalPopover feature="hrf" placement="bottom">
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
      <IAISwitch
        label="Toggle High Resolution Fix"
        isChecked={hrfToggled}
        onChange={handleHrfToggle}
      />
    </IAIInformationalPopover>
  );
}
