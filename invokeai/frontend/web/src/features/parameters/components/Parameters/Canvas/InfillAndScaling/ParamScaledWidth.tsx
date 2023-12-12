import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ canvas, generation }) => {
    const { boundingBoxScaleMethod, scaledBoundingBoxDimensions } = canvas;
    const { model, aspectRatio } = generation;

    return {
      model,
      scaledBoundingBoxDimensions,
      aspectRatio,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  }
);

const ParamScaledWidth = () => {
  const dispatch = useAppDispatch();
  const { model, isManual, scaledBoundingBoxDimensions, aspectRatio } =
    useAppSelector(selector);

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

  const { t } = useTranslation();

  const handleChangeScaledWidth = useCallback(
    (v: number) => {
      const newWidth = Math.floor(v);
      let newHeight = scaledBoundingBoxDimensions.height;

      if (aspectRatio) {
        newHeight = roundToMultiple(newWidth / aspectRatio, 64);
      }

      dispatch(
        setScaledBoundingBoxDimensions({
          width: newWidth,
          height: newHeight,
        })
      );
    },
    [aspectRatio, dispatch, scaledBoundingBoxDimensions.height]
  );

  const handleResetScaledWidth = useCallback(() => {
    const resetWidth = Math.floor(initial);
    let resetHeight = scaledBoundingBoxDimensions.height;

    if (aspectRatio) {
      resetHeight = roundToMultiple(resetWidth / aspectRatio, 64);
    }

    dispatch(
      setScaledBoundingBoxDimensions({
        width: resetWidth,
        height: resetHeight,
      })
    );
  }, [aspectRatio, dispatch, initial, scaledBoundingBoxDimensions.height]);

  return (
    <IAISlider
      isDisabled={!isManual}
      label={t('parameters.scaledWidth')}
      min={64}
      max={1536}
      step={64}
      value={scaledBoundingBoxDimensions.width}
      onChange={handleChangeScaledWidth}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleResetScaledWidth}
    />
  );
};

export default memo(ParamScaledWidth);
