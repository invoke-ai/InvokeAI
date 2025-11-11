import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/paramsSlice';
import { selectActiveCanvas } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.scaleMethod === 'manual');
const selectScaledWidth = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.scaledSize.width);

const CONSTRAINTS = {
  initial: 512,
  sliderMin: 64,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 4096,
  fineStep: 8,
  coarseStep: 64,
};

const BboxScaledWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledWidth = useAppSelector(selectScaledWidth);
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
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
        marks
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
        value={scaledWidth}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(BboxScaledWidth);
