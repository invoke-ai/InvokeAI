import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setClipSkip } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamClipSkip() {
  const clipSkip = useAppSelector(
    (state: RootState) => state.generation.clipSkip
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleClipSkipChange = useCallback(
    (v: number) => {
      dispatch(setClipSkip(v));
    },
    [dispatch]
  );

  const handleClipSkipReset = useCallback(() => {
    dispatch(setClipSkip(0));
  }, [dispatch]);

  return (
    <IAISlider
      label={t('parameters.clipSkip')}
      aria-label={t('parameters.clipSkip')}
      min={0}
      max={30}
      step={1}
      value={clipSkip}
      onChange={handleClipSkipChange}
      withSliderMarks
      sliderMarks={[0, 1, 2, 3, 5, 10, 15, 25, 30]}
      withInput
      withReset
      handleReset={handleClipSkipReset}
    />
  );
}
