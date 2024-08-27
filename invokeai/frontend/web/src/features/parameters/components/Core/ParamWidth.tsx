import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectWidth = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.rect.width);
const selectWidthConfig = createSelector(selectConfigSlice, (config) => config.sd.width);

export const ParamWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector(selectWidth);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const config = useAppSelector(selectWidthConfig);

  const onChange = useCallback(
    (v: number) => {
      dispatch(bboxWidthChanged({ width: v }));
    },
    [dispatch]
  );

  const marks = useMemo(
    () => [config.sliderMin, optimalDimension, config.sliderMax],
    [config.sliderMax, config.sliderMin, optimalDimension]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramWidth">
        <FormLabel>{t('parameters.width')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        marks={marks}
      />
      <CompositeNumberInput
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
      />
    </FormControl>
  );
});

ParamWidth.displayName = 'ParamWidth';
