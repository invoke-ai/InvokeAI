import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectGuidance, setGuidance } from 'features/controlLayers/store/paramsSlice';
import { selectGuidanceConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamGuidance = () => {
  const guidance = useAppSelector(selectGuidance);
  const config = useAppSelector(selectGuidanceConfig);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(
    () => [
      config.sliderMin,
      Math.floor(config.sliderMax - (config.sliderMax - config.sliderMin) / 2),
      config.sliderMax,
    ],
    [config.sliderMax, config.sliderMin]
  );
  const onChange = useCallback((v: number) => dispatch(setGuidance(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramGuidance">
        <FormLabel>{t('parameters.guidance')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={guidance}
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={guidance}
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

export default memo(ParamGuidance);
