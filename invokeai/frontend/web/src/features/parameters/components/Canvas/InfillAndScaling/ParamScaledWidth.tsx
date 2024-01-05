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

const ParamScaledWidth = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(
    (s) => s.canvas.boundingBoxScaleMethod === 'manual'
  );
  const width = useAppSelector(
    (s) => s.canvas.scaledBoundingBoxDimensions.width
  );

  const onChange = useCallback(
    (width: number) => {
      dispatch(setScaledBoundingBoxDimensions({ width }));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(setScaledBoundingBoxDimensions({ width: optimalDimension }));
  }, [dispatch, optimalDimension]);

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledWidth')}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={width}
        onChange={onChange}
        numberInputMax={4096}
        marks
        withNumberInput
        onReset={onReset}
      />
    </InvControl>
  );
};

export default memo(ParamScaledWidth);
