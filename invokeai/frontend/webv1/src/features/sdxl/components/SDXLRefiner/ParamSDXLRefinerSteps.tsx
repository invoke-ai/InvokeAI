import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectRefinerSteps, setRefinerSteps } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 30,
  sliderMin: 1,
  sliderMax: 100,
  numberInputMin: 1,
  numberInputMax: 500,
  fineStep: 1,
  coarseStep: 1,
};

const MARKS = [CONSTRAINTS.sliderMin, Math.floor(CONSTRAINTS.sliderMax / 2), CONSTRAINTS.sliderMax];

const ParamSDXLRefinerSteps = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const refinerSteps = useAppSelector(selectRefinerSteps);

  const onChange = useCallback(
    (v: number) => {
      dispatch(setRefinerSteps(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="refinerSteps">
        <FormLabel>{t('sdxl.steps')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={refinerSteps}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={refinerSteps}
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

export default memo(ParamSDXLRefinerSteps);
