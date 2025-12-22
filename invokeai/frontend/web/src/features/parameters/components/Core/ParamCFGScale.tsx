import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCFGScale, setCfgScale } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 7,
  sliderMin: 0,
  sliderMax: 20,
  numberInputMin: 0,
  numberInputMax: 200,
  fineStep: 0.1,
  coarseStep: 0.5,
};

export const MARKS = [CONSTRAINTS.sliderMin, Math.floor(CONSTRAINTS.sliderMax / 2), CONSTRAINTS.sliderMax];

const ParamCFGScale = () => {
  const cfgScale = useAppSelector(selectCFGScale);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback((v: number) => dispatch(setCfgScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramCFGScale">
        <FormLabel>{t('parameters.cfgScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={cfgScale}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={cfgScale}
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

export default memo(ParamCFGScale);
