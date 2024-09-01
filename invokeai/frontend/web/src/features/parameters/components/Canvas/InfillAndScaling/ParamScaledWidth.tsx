import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaleMethod === 'manual');
const selectScaledWidth = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaledSize.width);
const selectScaledBoundingBoxWidthConfig = createSelector(
  selectConfigSlice,
  (config) => config.sd.scaledBoundingBoxWidth
);

const ParamScaledWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledWidth = useAppSelector(selectScaledWidth);
  const config = useAppSelector(selectScaledBoundingBoxWidthConfig);
  const onChange = useCallback(
    (width: number) => {
      dispatch(bboxScaledWidthChanged(width));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual}>
      <FormLabel>{t('parameters.scaledWidth')}</FormLabel>
      <CompositeSlider
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
        marks
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(ParamScaledWidth);
