import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCfgScale } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGScale = () => {
  const cfgScale = useAppSelector((s) => s.generation.cfgScale);
  const sliderMin = useAppSelector((s) => s.config.sd.guidance.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.guidance.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.guidance.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.guidance.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.guidance.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.guidance.fineStep);
  const initial = useAppSelector((s) => s.config.sd.guidance.initial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, Math.floor(sliderMax / 2), sliderMax], [sliderMax, sliderMin]);
  const onChange = useCallback((v: number) => dispatch(setCfgScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramCFGScale">
        <FormLabel>{t('parameters.cfgScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={cfgScale}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={cfgScale}
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

export default memo(ParamCFGScale);
