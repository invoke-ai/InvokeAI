import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledHeightChanged } from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSlice, selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectCanvasSlice, (canvas) => canvas ? canvas.bbox.scaleMethod === 'manual' : false);
const selectScaledHeight = createSelector(selectCanvasSlice, (canvas) => canvas ? canvas.bbox.scaledSize.height : 512);
const selectScaledBoundingBoxHeightConfig = createSelector(
  selectConfigSlice,
  (config) => config.sd.scaledBoundingBoxHeight
);

const BboxScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledHeight = useAppSelector(selectScaledHeight);
  const config = useAppSelector(selectScaledBoundingBoxHeightConfig);
  const gridSize = useAppSelector(selectGridSize);

  const onChange = useCallback(
    (height: number) => {
      dispatch(bboxScaledHeightChanged(height));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual || isBboxSizeLocked}>
      <FormLabel>{t('parameters.scaledHeight')}</FormLabel>
      <CompositeSlider
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={gridSize}
        value={scaledHeight}
        onChange={onChange}
        marks
        defaultValue={optimalDimension}
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={gridSize}
        value={scaledHeight}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(BboxScaledHeight);
