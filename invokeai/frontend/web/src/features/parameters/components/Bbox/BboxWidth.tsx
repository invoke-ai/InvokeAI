import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { selectGridSize, selectOptimalDimension, selectWidth } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 512,
  sliderMin: 64,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 4096,
  fineStep: 8,
  coarseStep: 64,
};

export const BboxWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector(selectWidth);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const gridSize = useAppSelector(selectGridSize);

  const onChange = useCallback(
    (v: number) => {
      dispatch(bboxWidthChanged({ width: v }));
    },
    [dispatch]
  );

  const marks = useMemo(() => [CONSTRAINTS.sliderMin, optimalDimension, CONSTRAINTS.sliderMax], [optimalDimension]);

  return (
    <FormControl isDisabled={isBboxSizeLocked}>
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

BboxWidth.displayName = 'BboxWidth';
