import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCFGScale, setCfgScale } from 'features/controlLayers/store/paramsSlice';
import { selectCFGScaleConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGScale = () => {
  const cfgScale = useAppSelector(selectCFGScale);
  const config = useAppSelector(selectCFGScaleConfig);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(
    () => [config.sliderMin, Math.floor(config.sliderMax / 2), config.sliderMax],
    [config.sliderMax, config.sliderMin]
  );
  const onChange = useCallback((v: number) => dispatch(setCfgScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramCFGScale">
        <FormLabel>{t('parameters.cfgScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={cfgScale}
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={cfgScale}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamCFGScale);
