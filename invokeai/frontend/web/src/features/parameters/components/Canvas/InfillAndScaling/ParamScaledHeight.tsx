import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxScaledSizeChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectIsManual = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaleMethod === 'manual');
const selectScaledHeight = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.scaledSize.height);
const selectScaledBoundingBoxHeightConfig = createSelector(
  selectConfigSlice,
  (config) => config.sd.scaledBoundingBoxHeight
);

const ParamScaledHeight = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isManual = useAppSelector(selectIsManual);
  const scaledHeight = useAppSelector(selectScaledHeight);
  const config = useAppSelector(selectScaledBoundingBoxHeightConfig);

  const onChange = useCallback(
    (height: number) => {
      dispatch(bboxScaledSizeChanged({ height }));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!isManual}>
      <FormLabel>{t('parameters.scaledHeight')}</FormLabel>
      <CompositeSlider
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={scaledHeight}
        onChange={onChange}
        marks
        defaultValue={optimalDimension}
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={scaledHeight}
        onChange={onChange}
        defaultValue={optimalDimension}
      />
    </FormControl>
  );
};

export default memo(ParamScaledHeight);
