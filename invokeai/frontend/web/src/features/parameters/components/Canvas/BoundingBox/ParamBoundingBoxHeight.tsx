import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ canvas, generation }, isStaging) => {
    const { boundingBoxDimensions } = canvas;
    const { model, aspectRatio } = generation;
    return {
      model,
      boundingBoxDimensions,
      isStaging,
      aspectRatio,
    };
  }
);

const ParamBoundingBoxWidth = () => {
  const dispatch = useAppDispatch();
  const { model, boundingBoxDimensions, isStaging, aspectRatio } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

  const handleChangeHeight = useCallback(
    (v: number) => {
      dispatch(
        setBoundingBoxDimensions({
          ...boundingBoxDimensions,
          height: Math.floor(v),
        })
      );
      if (aspectRatio) {
        const newWidth = roundToMultiple(v * aspectRatio.value, 64);
        dispatch(
          setBoundingBoxDimensions({
            width: newWidth,
            height: Math.floor(v),
          })
        );
      }
    },
    [aspectRatio, boundingBoxDimensions, dispatch]
  );

  const handleResetHeight = useCallback(() => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(initial),
      })
    );
    if (aspectRatio) {
      const newWidth = roundToMultiple(initial * aspectRatio.value, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: newWidth,
          height: Math.floor(initial),
        })
      );
    }
  }, [aspectRatio, boundingBoxDimensions, dispatch, initial]);

  return (
    <InvControl label={t('parameters.height')} isDisabled={isStaging}>
      <InvSlider
        min={64}
        max={1536}
        step={64}
        value={boundingBoxDimensions.height}
        onChange={handleChangeHeight}
        onReset={handleResetHeight}
        marks
        withNumberInput
        numberInputMax={4096}
      />
    </InvControl>
  );
};

export default memo(ParamBoundingBoxWidth);
