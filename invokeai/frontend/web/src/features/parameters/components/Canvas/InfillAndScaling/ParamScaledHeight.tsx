import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(
    (s) => s.canvas.boundingBoxScaleMethod === 'manual'
  );
  const height = useAppSelector(
    (s) => s.canvas.scaledBoundingBoxDimensions.height
  );
  const sliderMin = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.coarseStep
  );
  const fineStep = useAppSelector(
    (s) => s.config.sd.scaledBoundingBoxHeight.fineStep
  );

  const onChange = useCallback(
    (height: number) => {
      dispatch(setScaledBoundingBoxDimensions({ height }));
    },
    [dispatch]
  );

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledHeight')}>
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={height}
        onChange={onChange}
        marks
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        defaultValue={optimalDimension}
      />
    </InvControl>
  );
};

export default memo(ParamScaledHeight);
