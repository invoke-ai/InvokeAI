import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { scaledBboxChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector((s) => s.canvasV2.bbox.scaleMethod === 'manual');
  const height = useAppSelector((s) => s.canvasV2.bbox.scaledSize.height);
  const sliderMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.fineStep);

  const onChange = useCallback(
    (height: number) => {
      dispatch(scaledBboxChanged({ height }));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual}>
      <FormLabel>{t('parameters.scaledHeight')}</FormLabel>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={height}
        onChange={onChange}
        marks
        defaultValue={optimalDimension}
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={height}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(ParamScaledHeight);
