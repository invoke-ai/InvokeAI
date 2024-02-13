import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setHrfStrength } from 'features/hrf/store/hrfSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamHrfStrength = () => {
  const hrfStrength = useAppSelector((s) => s.hrf.hrfStrength);
  const initial = useAppSelector((s) => s.config.sd.hrfStrength.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.hrfStrength.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.hrfStrength.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.hrfStrength.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.hrfStrength.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.hrfStrength.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.hrfStrength.fineStep);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('parameters.denoisingStrength')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={hrfStrength}
        defaultValue={initial}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={hrfStrength}
        defaultValue={initial}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamHrfStrength);
