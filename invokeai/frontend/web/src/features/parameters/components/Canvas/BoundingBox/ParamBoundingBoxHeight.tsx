import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBoundingBoxHeight = () => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const isStaging = useAppSelector(isStagingSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector((s) => s.config.sd.boundingBoxHeight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.boundingBoxHeight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.boundingBoxHeight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.boundingBoxHeight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.boundingBoxHeight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.boundingBoxHeight.fineStep);
  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  return (
    <FormControl isDisabled={isStaging}>
      <FormLabel>{t('parameters.height')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamBoundingBoxHeight);
