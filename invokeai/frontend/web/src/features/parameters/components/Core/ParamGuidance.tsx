import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectGuidance, selectGuidanceControl, setGuidance } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 4,
  sliderMin: 1,
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
  const externalControl = useAppSelector(selectGuidanceControl);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setGuidance(v)), [dispatch]);

  const sliderMin = externalControl?.slider_min ?? CONSTRAINTS.sliderMin;
  const sliderMax = externalControl?.slider_max ?? CONSTRAINTS.sliderMax;
  const numberInputMin = externalControl?.number_input_min ?? CONSTRAINTS.numberInputMin;
  const numberInputMax = externalControl?.number_input_max ?? CONSTRAINTS.numberInputMax;
  const fineStep = externalControl?.fine_step ?? CONSTRAINTS.fineStep;
  const coarseStep = externalControl?.coarse_step ?? CONSTRAINTS.coarseStep;
  const marks = useMemo(
    () => externalControl?.marks ?? [sliderMin, Math.floor(sliderMax - (sliderMax - sliderMin) / 2), sliderMax],
    [externalControl?.marks, sliderMin, sliderMax]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramGuidance">
        <FormLabel>{t('parameters.guidance')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={guidance}
        defaultValue={CONSTRAINTS.initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={guidance}
        defaultValue={CONSTRAINTS.initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamGuidance);
