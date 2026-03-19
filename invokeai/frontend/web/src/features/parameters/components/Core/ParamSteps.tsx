import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectSteps, selectStepsControl, setSteps } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 30,
  sliderMin: 1,
  sliderMax: 100,
  numberInputMin: 1,
  numberInputMax: 500,
  fineStep: 1,
  coarseStep: 1,
};

export const MARKS = [CONSTRAINTS.sliderMin, Math.floor(CONSTRAINTS.sliderMax / 2), CONSTRAINTS.sliderMax];

const ParamSteps = () => {
  const steps = useAppSelector(selectSteps);
  const externalControl = useAppSelector(selectStepsControl);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (v: number) => {
      dispatch(setSteps(v));
    },
    [dispatch]
  );

  const sliderMin = externalControl?.slider_min ?? CONSTRAINTS.sliderMin;
  const sliderMax = externalControl?.slider_max ?? CONSTRAINTS.sliderMax;
  const numberInputMin = externalControl?.number_input_min ?? CONSTRAINTS.numberInputMin;
  const numberInputMax = externalControl?.number_input_max ?? CONSTRAINTS.numberInputMax;
  const fineStep = externalControl?.fine_step ?? CONSTRAINTS.fineStep;
  const coarseStep = externalControl?.coarse_step ?? CONSTRAINTS.coarseStep;
  const marks = useMemo(
    () => externalControl?.marks ?? [sliderMin, Math.floor(sliderMax / 2), sliderMax],
    [externalControl?.marks, sliderMin, sliderMax]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramSteps">
        <FormLabel>{t('parameters.steps')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={steps}
        defaultValue={CONSTRAINTS.initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={steps}
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

export default memo(ParamSteps);
