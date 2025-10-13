import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectGuidance, setGuidance } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 4,
  sliderMin: 2,
  sliderMax: 6,
  numberInputMin: 1,
  numberInputMax: 20,
  fineStep: 0.1,
  coarseStep: 0.5,
};

export const MARKS = [
  CONSTRAINTS.sliderMin,
  Math.floor(CONSTRAINTS.sliderMax - (CONSTRAINTS.sliderMax - CONSTRAINTS.sliderMin) / 2),
  CONSTRAINTS.sliderMax,
];

const ParamGuidance = () => {
  const guidance = useAppSelector(selectGuidance);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setGuidance(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramGuidance">
        <FormLabel>{t('parameters.guidance')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={guidance}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={guidance}
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

export default memo(ParamGuidance);
