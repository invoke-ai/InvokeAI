import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledWidthChanged } from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSlice, selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectCanvasSlice, (canvas) => canvas ? canvas.bbox.scaleMethod === 'manual' : false);
const selectScaledWidth = createSelector(selectCanvasSlice, (canvas) => canvas ? canvas.bbox.scaledSize.width : 512);
const selectScaledBoundingBoxWidthConfig = createSelector(
  selectConfigSlice,
  (config) => config.sd.scaledBoundingBoxWidth
);

const BboxScaledWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledWidth = useAppSelector(selectScaledWidth);
  const config = useAppSelector(selectScaledBoundingBoxWidthConfig);
  const gridSize = useAppSelector(selectGridSize);

  const onChange = useCallback(
    (width: number) => {
      dispatch(bboxScaledWidthChanged(width));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual || isBboxSizeLocked}>
      <FormLabel>{t('parameters.scaledWidth')}</FormLabel>
      <CompositeSlider
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={gridSize}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
        marks
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={gridSize}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(BboxScaledWidth);
