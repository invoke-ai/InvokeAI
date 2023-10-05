import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setHrf } from 'features/parameters/store/generationSlice';
import { clipSkipMap } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamHrf() {
  const hrfScale = useAppSelector(
    (state: RootState) => state.generation.hrfScale
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
        label={t('parameters.hrf')}
        aria-label={t('parameters.hrf')}
        min={0}
        max={20}
        step={1}
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
