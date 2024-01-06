import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import {
  CANVAS_GRID_SIZE_COARSE,
  CANVAS_GRID_SIZE_FINE,
} from 'features/canvas/store/constants';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaledHeight = () => {
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(
    (s) => s.canvas.boundingBoxScaleMethod === 'manual'
  );
  const height = useAppSelector(
    (s) => s.canvas.scaledBoundingBoxDimensions.height
  );

  const { t } = useTranslation();

  const onChange = useCallback(
    (height: number) => {
      dispatch(setScaledBoundingBoxDimensions({ height }));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(setScaledBoundingBoxDimensions({ height: optimalDimension }));
  }, [dispatch, optimalDimension]);

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledHeight')}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={height}
        onChange={onChange}
        marks
        withNumberInput
        numberInputMax={4096}
        onReset={onReset}
      />
    </InvControl>
  );
};

export default memo(ParamScaledHeight);
