import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBoundingBoxWidth = () => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const isStaging = useAppSelector(isStagingSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector((s) => s.config.sd.boundingBoxWidth.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.boundingBoxWidth.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.boundingBoxWidth.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.boundingBoxWidth.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.boundingBoxWidth.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.boundingBoxWidth.fineStep);
  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  return (
    <FormControl isDisabled={isStaging}>
      <FormLabel>{t('parameters.width')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.width}
        defaultValue={optimalDimension}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.width}
        defaultValue={optimalDimension}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamBoundingBoxWidth);
