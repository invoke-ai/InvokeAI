import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxWidthChanged } from 'features/controlLayers/store/canvasInstanceSlice';
import { selectGridSize, selectOptimalDimension, selectWidth } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { selectWidthConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const BboxWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector(selectWidth);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const config = useAppSelector(selectWidthConfig);
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const gridSize = useAppSelector(selectGridSize);

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
    <FormControl isDisabled={isBboxSizeLocked}>
      <InformationalPopover feature="paramWidth">
        <FormLabel>{t('parameters.width')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={width ?? optimalDimension}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={gridSize}
        marks={marks}
      />
      <CompositeNumberInput
        value={width ?? optimalDimension}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={gridSize}
      />
    </FormControl>
  );
});

BboxWidth.displayName = 'BboxWidth';
