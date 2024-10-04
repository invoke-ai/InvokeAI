import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledHeightChanged } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectCanvasSlice, selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaleMethod === 'manual');
const selectScaledHeight = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaledSize.height);
const selectScaledBoundingBoxHeightConfig = createSelector(
  selectConfigSlice,
  (config) => config.sd.scaledBoundingBoxHeight
);

const BboxScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(selectIsStaging);
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
    <FormControl isDisabled={!isManual || isStaging}>
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
