import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setClipSkip } from 'features/parameters/store/generationSlice';
import { clipSkipMap } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamClipSkip() {
  const clipSkip = useAppSelector(
    (state: RootState) => state.generation.clipSkip
  );

  const { model } = useAppSelector((state: RootState) => state.generation);

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

  const max = useMemo(() => {
    if (!model) {
      return clipSkipMap['sd-1'].maxClip;
    }
    return clipSkipMap[model.base_model].maxClip;
  }, [model]);

  const sliderMarks = useMemo(() => {
    if (!model) {
      return clipSkipMap['sd-1'].markers;
    }
    return clipSkipMap[model.base_model].markers;
  }, [model]);

  return (
    <IAISlider
      label={t('parameters.clipSkip')}
      aria-label={t('parameters.clipSkip')}
      min={0}
      max={max}
      step={1}
      value={clipSkip}
      onChange={handleClipSkipChange}
      withSliderMarks
      sliderMarks={sliderMarks}
      withInput
      withReset
      handleReset={handleClipSkipReset}
    />
  );
}
