import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledHeightChanged } from 'features/controlLayers/store/canvasSlice';
import { selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/paramsSlice';
import { selectActiveCanvas } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.scaleMethod === 'manual');
const selectScaledHeight = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.scaledSize.height);

const CONSTRAINTS = {
  initial: 512,
  sliderMin: 64,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 4096,
  fineStep: 8,
  coarseStep: 64,
};

const BboxScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledHeight = useAppSelector(selectScaledHeight);
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
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
        value={scaledHeight}
        onChange={onChange}
        marks
        defaultValue={optimalDimension}
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
        value={scaledHeight}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(BboxScaledHeight);
