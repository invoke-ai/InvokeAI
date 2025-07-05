import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectWidth, widthChanged } from 'features/controlLayers/store/paramsSlice';
import { selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';
import { selectWidthConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const DimensionsWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector(selectWidth);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const config = useAppSelector(selectWidthConfig);
  const isApiModel = useIsApiModel();
  const gridSize = useAppSelector(selectGridSize);

  const onChange = useCallback(
    (v: number) => {
      dispatch(widthChanged({ width: v }));
    },
    [dispatch]
  );

  const marks = useMemo(
    () => [config.sliderMin, optimalDimension, config.sliderMax],
    [config.sliderMax, config.sliderMin, optimalDimension]
  );

  return (
    <FormControl isDisabled={isApiModel}>
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
        fineStep={gridSize}
        marks={marks}
      />
      <CompositeNumberInput
        value={width}
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

DimensionsWidth.displayName = 'Dimensions';
