import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectSteps, setSteps } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
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
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (v: number) => {
      dispatch(setSteps(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramSteps">
        <FormLabel>{t('parameters.steps')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={steps}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={steps}
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

export default memo(ParamSteps);
