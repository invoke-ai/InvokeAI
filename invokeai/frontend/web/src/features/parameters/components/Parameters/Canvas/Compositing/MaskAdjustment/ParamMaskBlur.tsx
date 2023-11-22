import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setMaskBlur } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamMaskBlur() {
  const dispatch = useAppDispatch();
  const maskBlur = useAppSelector(
    (state: RootState) => state.generation.maskBlur
  );
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setMaskBlur(v));
    },
    [dispatch]
  );
  const handleReset = useCallback(() => {
    dispatch(setMaskBlur(16));
  }, [dispatch]);

  return (
    <IAIInformationalPopover feature="compositingBlur">
      <IAISlider
        label={t('parameters.maskBlur')}
        min={0}
        max={64}
        sliderNumberInputProps={{ max: 512 }}
        value={maskBlur}
        onChange={handleChange}
        withInput
        withSliderMarks
        withReset
        handleReset={handleReset}
      />
    </IAIInformationalPopover>
  );
}
