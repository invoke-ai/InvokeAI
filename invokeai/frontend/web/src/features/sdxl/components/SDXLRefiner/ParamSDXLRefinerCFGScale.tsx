import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectRefinerCFGScale, setRefinerCFGScale } from 'features/controlLayers/store/paramsSlice';
import { selectCFGScaleConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerCFGScale = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const refinerCFGScale = useAppSelector(selectRefinerCFGScale);
  const config = useAppSelector(selectCFGScaleConfig);
  const marks = useMemo(
    () => [config.sliderMin, Math.floor(config.sliderMax / 2), config.sliderMax],
    [config.sliderMax, config.sliderMin]
  );

  const onChange = useCallback((v: number) => dispatch(setRefinerCFGScale(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="refinerCfgScale">
        <FormLabel>{t('sdxl.cfgScale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={refinerCFGScale}
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={refinerCFGScale}
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

export default memo(ParamSDXLRefinerCFGScale);
