import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCFGRescaleMultiplier, setCfgRescaleMultiplier } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 0,
  sliderMin: 0,
  sliderMax: 0.99,
  numberInputMin: 0,
  numberInputMax: 0.99,
  fineStep: 0.05,
  coarseStep: 0.1,
};

const ParamCFGRescaleMultiplier = () => {
  const cfgRescaleMultiplier = useAppSelector(selectCFGRescaleMultiplier);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback((v: number) => dispatch(setCfgRescaleMultiplier(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramCFGRescaleMultiplier">
        <FormLabel>{t('parameters.cfgRescaleMultiplier')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={cfgRescaleMultiplier}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        value={cfgRescaleMultiplier}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCFGRescaleMultiplier);
