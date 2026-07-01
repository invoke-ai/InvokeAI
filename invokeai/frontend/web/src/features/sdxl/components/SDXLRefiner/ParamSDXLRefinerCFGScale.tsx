import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectRefinerCFGScale, setRefinerCFGScale } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 7,
  sliderMin: 1,
  sliderMax: 20,
  numberInputMin: 1,
  numberInputMax: 200,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [CONSTRAINTS.sliderMin, Math.floor(CONSTRAINTS.sliderMax / 2), CONSTRAINTS.sliderMax];

const ParamSDXLRefinerCFGScale = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const refinerCFGScale = useAppSelector(selectRefinerCFGScale);

  const onChange = useCallback((v: number) => dispatch(setRefinerCFGScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="refinerCfgScale">
        <FormLabel>{t('sdxl.cfgScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={refinerCFGScale}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={refinerCFGScale}
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

export default memo(ParamSDXLRefinerCFGScale);
