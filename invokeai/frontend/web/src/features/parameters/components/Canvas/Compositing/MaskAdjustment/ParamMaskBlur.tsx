import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectMaskBlur, setMaskBlur, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import { selectMaskBlurConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamMaskBlur = () => {
  const { t } = useTranslation();
  const dispatchParams = useParamsDispatch();
  const maskBlur = useAppSelector(selectMaskBlur);
  const config = useAppSelector(selectMaskBlurConfig);

  const handleChange = useCallback(
    (v: number) => {
      dispatchParams(setMaskBlur, v);
    },
    [dispatchParams]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingMaskBlur">
        <FormLabel>{t('parameters.maskBlur')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={maskBlur}
        onChange={handleChange}
        min={config.sliderMin}
        max={config.sliderMax}
        defaultValue={config.initial}
        step={config.coarseStep}
        fineStep={config.fineStep}
        marks
      />
      <CompositeNumberInput
        value={maskBlur}
        onChange={handleChange}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamMaskBlur);
