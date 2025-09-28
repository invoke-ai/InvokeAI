import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  heightChanged,
  selectGridSize,
  selectHeight,
  selectIsApiBaseModel,
  selectOptimalDimension,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { selectHeightConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const DimensionsHeight = memo(() => {
  const { t } = useTranslation();
  const dispatchParams = useParamsDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const height = useAppSelector(selectHeight);
  const config = useAppSelector(selectHeightConfig);
  const gridSize = useAppSelector(selectGridSize);
  const isApiModel = useAppSelector(selectIsApiBaseModel);

  const onChange = useCallback(
    (v: number) => {
      dispatchParams(heightChanged, { height: v });
    },
    [dispatchParams]
  );

  const marks = useMemo(
    () => [config.sliderMin, optimalDimension, config.sliderMax],
    [config.sliderMin, config.sliderMax, optimalDimension]
  );

  return (
    <FormControl isDisabled={isApiModel}>
      <InformationalPopover feature="paramHeight">
        <FormLabel>{t('parameters.height')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={gridSize}
        marks={marks}
      />
      <CompositeNumberInput
        value={height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={gridSize}
      />
    </FormControl>
  );
});

DimensionsHeight.displayName = 'DimensionsHeight';
