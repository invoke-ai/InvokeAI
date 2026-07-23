import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectKrea2SeedVarianceStrength,
  setKrea2SeedVarianceStrength,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 20,
  sliderMin: 0,
  sliderMax: 50,
  numberInputMin: 0,
  numberInputMax: 100,
  fineStep: 1,
  coarseStep: 5,
};

const MARKS = [0, 10, 20, 30, 40, 50];

const ParamKrea2SeedVarianceStrength = () => {
  const strength = useAppSelector(selectKrea2SeedVarianceStrength);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setKrea2SeedVarianceStrength(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="krea2SeedVarianceStrength">
        <FormLabel>{t('parameters.seedVarianceStrength')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={strength}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={strength}
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

export default memo(ParamKrea2SeedVarianceStrength);
