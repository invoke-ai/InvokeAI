import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectZImageSeedVarianceRandomizePercent,
  setZImageSeedVarianceRandomizePercent,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 50,
  sliderMin: 1,
  sliderMax: 100,
  numberInputMin: 1,
  numberInputMax: 100,
  fineStep: 1,
  coarseStep: 5,
};

const MARKS = [1, 25, 50, 75, 100];

const ParamZImageSeedVarianceRandomizePercent = () => {
  const randomizePercent = useAppSelector(selectZImageSeedVarianceRandomizePercent);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setZImageSeedVarianceRandomizePercent(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="seedVarianceRandomizePercent">
        <FormLabel>{t('parameters.seedVarianceRandomizePercent')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={randomizePercent}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={randomizePercent}
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

export default memo(ParamZImageSeedVarianceRandomizePercent);
