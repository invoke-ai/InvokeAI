import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCfgRescaleMultiplier } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGRescaleMultiplier = () => {
  const cfgRescaleMultiplier = useAppSelector((s) => s.generation.cfgRescaleMultiplier);
  const initial = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.cfgRescaleMultiplier.fineStep);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback((v: number) => dispatch(setCfgRescaleMultiplier(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramCFGRescaleMultiplier">
        <FormLabel>{t('parameters.cfgRescaleMultiplier')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={cfgRescaleMultiplier}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        value={cfgRescaleMultiplier}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCFGRescaleMultiplier);
