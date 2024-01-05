import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
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

const selector = createMemoizedSelector(
  [stateSelector, selectOptimalDimension],
  ({ canvas }, optimalDimension) => {
    const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = canvas;

    return {
      optimalDimension,
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  }
);

const ParamScaledHeight = () => {
  const dispatch = useAppDispatch();
  const { isManual, scaledBoundingBoxDimensions, optimalDimension } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeScaledHeight = useCallback(
    (height: number) => {
      dispatch(setScaledBoundingBoxDimensions({ height }));
    },
    [dispatch]
  );

  const handleResetScaledHeight = useCallback(() => {
    dispatch(setScaledBoundingBoxDimensions({ height: optimalDimension }));
  }, [dispatch, optimalDimension]);

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledHeight')}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={scaledBoundingBoxDimensions.height}
        onChange={handleChangeScaledHeight}
        marks
        withNumberInput
        numberInputMax={4096}
        onReset={handleResetScaledHeight}
      />
    </InvControl>
  );
};

export default memo(ParamScaledHeight);
