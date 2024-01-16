import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(
    (s) => s.canvas.boundingBoxScaleMethod === 'manual'
  );
  const width = useAppSelector(
    (s) => s.canvas.scaledBoundingBoxDimensions.width
  );
  const sliderMin = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.coarseStep
  );
  const fineStep = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxWidth.fineStep
  );
  const onChange = useCallback(
    (width: number) => {
      dispatch(setScaledBoundingBoxDimensions({ width }));
    },
    [dispatch]
  );

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledWidth')}>
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={width}
        onChange={onChange}
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        defaultValue={optimalDimension}
        marks
        withNumberInput
      />
    </InvControl>
  );
};

export default memo(ParamScaledWidth);
