import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectKrea2RebalanceMultiplier, setKrea2RebalanceMultiplier } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 4,
  sliderMin: 0,
  sliderMax: 10,
  numberInputMin: 0,
  numberInputMax: 20,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [0, 2.5, 5, 7.5, 10];

const ParamKrea2RebalanceMultiplier = () => {
  const multiplier = useAppSelector(selectKrea2RebalanceMultiplier);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setKrea2RebalanceMultiplier(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="krea2RebalanceMultiplier">
        <FormLabel>{t('parameters.krea2RebalanceMultiplier')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={multiplier}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={multiplier}
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

export default memo(ParamKrea2RebalanceMultiplier);
