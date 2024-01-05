import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import {
  selectCanvasSlice,
  setScaledBoundingBoxDimensions,
} from 'features/canvas/store/canvasSlice';
import {
  CANVAS_GRID_SIZE_COARSE,
  CANVAS_GRID_SIZE_FINE,
} from 'features/canvas/store/constants';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [selectCanvasSlice, selectOptimalDimension],
  (canvas, optimalDimension) => {
    const { boundingBoxScaleMethod, scaledBoundingBoxDimensions } = canvas;

    return {
      optimalDimension,
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  }
);

const ParamScaledWidth = () => {
  const dispatch = useAppDispatch();
  const { optimalDimension, isManual, scaledBoundingBoxDimensions } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeScaledWidth = useCallback(
    (width: number) => {
      dispatch(setScaledBoundingBoxDimensions({ width }));
    },
    [dispatch]
  );

  const handleResetScaledWidth = useCallback(() => {
    dispatch(setScaledBoundingBoxDimensions({ width: optimalDimension }));
  }, [dispatch, optimalDimension]);

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledWidth')}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={scaledBoundingBoxDimensions.width}
        onChange={handleChangeScaledWidth}
        numberInputMax={4096}
        marks
        withNumberInput
        onReset={handleResetScaledWidth}
      />
    </InvControl>
  );
};

export default memo(ParamScaledWidth);
