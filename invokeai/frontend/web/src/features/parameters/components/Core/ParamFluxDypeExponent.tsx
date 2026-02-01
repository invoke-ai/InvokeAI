import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectFluxDypeExponent, setFluxDypeExponent } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

// DyPE Exponent (λt) - Controls the strength of the dynamic effect over time
// 2.0: Recommended for 4K+ resolutions - aggressive schedule that transitions quickly
// 1.0: Good starting point for ~2K-3K resolutions
// 0.5: Gentler schedule for resolutions just above native
const CONSTRAINTS = {
  initial: 2.0,
  sliderMin: 0,
  sliderMax: 10,
  numberInputMin: 0,
  numberInputMax: 1000,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [0, 0.5, 1, 2, 5, 10];

const ParamFluxDypeExponent = () => {
  const fluxDypeExponent = useAppSelector(selectFluxDypeExponent);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setFluxDypeExponent(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="fluxDypeExponent">
        <FormLabel>{t('parameters.dypeExponent')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={fluxDypeExponent}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={fluxDypeExponent}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamFluxDypeExponent);
