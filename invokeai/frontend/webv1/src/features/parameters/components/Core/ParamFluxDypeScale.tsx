import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectFluxDypeScale, setFluxDypeScale } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

// DyPE Scale (λs) - Controls the magnitude of the DyPE modulation
// Higher values = stronger extrapolation for high-resolution generation
const CONSTRAINTS = {
  initial: 2.0,
  sliderMin: 0,
  sliderMax: 8,
  numberInputMin: 0,
  numberInputMax: 8,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [0, 2, 4, 6, 8];

const ParamFluxDypeScale = () => {
  const fluxDypeScale = useAppSelector(selectFluxDypeScale);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setFluxDypeScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="fluxDypeScale">
        <FormLabel>{t('parameters.dypeScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={fluxDypeScale}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={fluxDypeScale}
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

export default memo(ParamFluxDypeScale);
