import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, canvas }) => {
    const { scaledBoundingBoxDimensions, boundingBoxScaleMethod, aspectRatio } =
      canvas;
    const { model } = generation;

    return {
      model,
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
      aspectRatio,
    };
  }
);

const ParamScaledHeight = () => {
  const dispatch = useAppDispatch();
  const { model, isManual, scaledBoundingBoxDimensions, aspectRatio } =
    useAppSelector(selector);

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

  const { t } = useTranslation();

  const handleChangeScaledHeight = useCallback(
    (v: number) => {
      let newWidth = scaledBoundingBoxDimensions.width;
      const newHeight = Math.floor(v);

      if (aspectRatio) {
        newWidth = roundToMultiple(newHeight * aspectRatio.value, 64);
      }

      dispatch(
        setScaledBoundingBoxDimensions({
          width: newWidth,
          height: newHeight,
        })
      );
    },
    [aspectRatio, dispatch, scaledBoundingBoxDimensions.width]
  );

  const handleResetScaledHeight = useCallback(() => {
    let resetWidth = scaledBoundingBoxDimensions.width;
    const resetHeight = Math.floor(initial);

    if (aspectRatio) {
      resetWidth = roundToMultiple(resetHeight * aspectRatio.value, 64);
    }

    dispatch(
      setScaledBoundingBoxDimensions({
        width: resetWidth,
        height: resetHeight,
      })
    );
  }, [aspectRatio, dispatch, initial, scaledBoundingBoxDimensions.width]);

  return (
    <InvControl isDisabled={!isManual} label={t('parameters.scaledHeight')}>
      <InvSlider
        min={64}
        max={1536}
        step={64}
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
