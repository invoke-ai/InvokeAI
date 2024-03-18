import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector((s) => s.canvas.boundingBoxScaleMethod === 'manual');
  const height = useAppSelector((s) => s.canvas.scaledBoundingBoxDimensions.height);
  const sliderMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.scaledBoundingBoxHeight.fineStep);

  const onChange = useCallback(
    (height: number) => {
      dispatch(setScaledBoundingBoxDimensions({ height }));
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
