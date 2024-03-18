import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerSteps = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const refinerSteps = useAppSelector((s) => s.sdxl.refinerSteps);
  const initial = useAppSelector((s) => s.config.sd.steps.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.steps.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.steps.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.steps.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.steps.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.steps.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.steps.fineStep);

  const marks = useMemo(() => [sliderMin, Math.floor(sliderMax / 2), sliderMax], [sliderMax, sliderMin]);

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
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={refinerSteps}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerSteps);
