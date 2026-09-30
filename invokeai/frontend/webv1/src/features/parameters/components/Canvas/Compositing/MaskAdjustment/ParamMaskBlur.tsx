import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectMaskBlur, setMaskBlur } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 16,
  sliderMin: 0,
  sliderMax: 128,
  numberInputMin: 0,
  numberInputMax: 512,
  fineStep: 1,
  coarseStep: 1,
};

const ParamMaskBlur = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const maskBlur = useAppSelector(selectMaskBlur);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setMaskBlur(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingMaskBlur">
        <FormLabel>{t('parameters.maskBlur')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={maskBlur}
        onChange={handleChange}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        defaultValue={CONSTRAINTS.initial}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        marks
      />
      <CompositeNumberInput
        value={maskBlur}
        onChange={handleChange}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamMaskBlur);
