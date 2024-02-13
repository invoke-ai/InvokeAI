import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
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
  const numberInputMin = useAppSelector((s) => s.config.sd.maskBlur.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.maskBlur.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.maskBlur.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.maskBlur.fineStep);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setMaskBlur(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingBlur">
        <FormLabel>{t('parameters.maskBlur')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        value={maskBlur}
        defaultValue={initial}
        onChange={handleChange}
        step={coarseStep}
        fineStep={fineStep}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        value={maskBlur}
        defaultValue={initial}
        onChange={handleChange}
        step={coarseStep}
        fineStep={fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamMaskBlur);
