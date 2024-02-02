import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamHeight = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector((s) => s.config.sd.height.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.height.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.height.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.height.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.height.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.height.fineStep);

  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(() => [sliderMin, optimalDimension, sliderMax], [sliderMin, optimalDimension, sliderMax]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.height')}</FormLabel>
      <CompositeSlider
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
      />
      <CompositeNumberInput
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
      />
    </FormControl>
  );
});

ParamHeight.displayName = 'ParamHeight';
