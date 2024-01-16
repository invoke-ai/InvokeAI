import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setMaskBlur } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamMaskBlur = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const maskBlur = useAppSelector((s) => s.generation.maskBlur);
  const initial = useAppSelector((s) => s.config.sd.maskBlur.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.maskBlur.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.maskBlur.sliderMax);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.maskBlur.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.maskBlur.numberInputMax
  );
  const coarseStep = useAppSelector((s) => s.config.sd.maskBlur.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.maskBlur.fineStep);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setMaskBlur(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.maskBlur')} feature="compositingBlur">
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        value={maskBlur}
        defaultValue={initial}
        onChange={handleChange}
        marks
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
      />
    </InvControl>
  );
};

export default memo(ParamMaskBlur);
