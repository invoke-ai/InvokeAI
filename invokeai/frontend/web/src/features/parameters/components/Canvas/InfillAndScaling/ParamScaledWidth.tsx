import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { scaledBboxChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector((s) => s.canvasV2.bbox.scaleMethod === 'manual');
  const width = useAppSelector((s) => s.canvasV2.bbox.scaledSize.width);
  const sliderMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxWidth.fineStep);
  const onChange = useCallback(
    (width: number) => {
      dispatch(scaledBboxChanged({ width }));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual}>
      <FormLabel>{t('parameters.scaledWidth')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(ParamScaledWidth);
