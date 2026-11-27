import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectZImageSeedVarianceStrength,
  setZImageSeedVarianceStrength,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 0.1,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 2,
  fineStep: 0.01,
  coarseStep: 0.05,
};

const MARKS = [0, 0.25, 0.5, 0.75, 1];

const ParamZImageSeedVarianceStrength = () => {
  const strength = useAppSelector(selectZImageSeedVarianceStrength);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setZImageSeedVarianceStrength(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="seedVarianceStrength">
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

export default memo(ParamZImageSeedVarianceStrength);
