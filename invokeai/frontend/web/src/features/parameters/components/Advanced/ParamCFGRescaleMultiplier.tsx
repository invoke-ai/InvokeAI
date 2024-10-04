import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCFGRescaleMultiplier, setCfgRescaleMultiplier } from 'features/controlLayers/store/paramsSlice';
import { selectCFGRescaleMultiplierConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGRescaleMultiplier = () => {
  const cfgRescaleMultiplier = useAppSelector(selectCFGRescaleMultiplier);
  const config = useAppSelector(selectCFGRescaleMultiplierConfig);

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
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        value={cfgRescaleMultiplier}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCFGRescaleMultiplier);
