import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setHrfStrength } from 'features/hrf/store/hrfSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamHrfStrength = () => {
  const hrfStrength = useAppSelector((s) => s.hrf.hrfStrength);
  const initial = useAppSelector((s) => s.config.sd.hrfStrength.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.hrfStrength.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.hrfStrength.sliderMax);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.hrfStrength.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.hrfStrength.numberInputMax
  );
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
    <InvControl label={t('parameters.denoisingStrength')}>
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={hrfStrength}
        defaultValue={initial}
        onChange={onChange}
        marks
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
      />
    </InvControl>
  );
};

export default memo(ParamHrfStrength);
