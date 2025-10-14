import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectWidth, widthChanged } from 'features/controlLayers/store/paramsSlice';
import { selectGridSize, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CONSTRAINTS = {
  initial: 512,
  sliderMin: 64,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 4096,
  fineStep: 8,
  coarseStep: 64,
};

export const DimensionsWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector(selectWidth);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const gridSize = useAppSelector(selectGridSize);

  const onChange = useCallback(
    (v: number) => {
      dispatch(widthChanged({ width: v }));
    },
    [dispatch]
  );

  const marks = useMemo(() => [CONSTRAINTS.sliderMin, optimalDimension, CONSTRAINTS.sliderMax], [optimalDimension]);

  return (
    <FormControl>
      <InformationalPopover feature="paramWidth">
        <FormLabel>{t('parameters.width')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
        marks={marks}
      />
      <CompositeNumberInput
        value={width}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={gridSize}
      />
    </FormControl>
  );
});

DimensionsWidth.displayName = 'Dimensions';
