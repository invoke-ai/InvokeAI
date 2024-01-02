import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { CANVAS_GRID_SIZE_COARSE, CANVAS_GRID_SIZE_FINE } from 'features/canvas/store/canvasSlice';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ generation }, isStaging) => {
    const { model } = generation;
    const initial = ['sdxl', 'sdxl-refiner'].includes(
      model?.base_model as string
    )
      ? 1024
      : 512;
    return {
      initial,
      model,
      isStaging,
    };
  }
);

const ParamBoundingBoxWidth = () => {
  const { isStaging, initial } = useAppSelector(selector);
  const ctx = useImageSizeContext();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  const onReset = useCallback(() => {
    ctx.widthChanged(initial);
  }, [ctx, initial]);

  return (
    <InvControl label={t('parameters.width')} isDisabled={isStaging}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={ctx.width}
        onChange={onChange}
        onReset={onReset}
        withNumberInput
        numberInputMax={4096}
        marks
      />
    </InvControl>
  );
};

export default memo(ParamBoundingBoxWidth);
