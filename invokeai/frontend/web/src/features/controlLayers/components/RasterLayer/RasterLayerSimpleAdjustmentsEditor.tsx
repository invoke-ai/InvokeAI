import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsSimpleUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type AdjustmentSliderRowProps = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
};

const AdjustmentSliderRow = ({ label, value, onChange, min = -1, max = 1, step = 0.01 }: AdjustmentSliderRowProps) => (
  <FormControl orientation="horizontal" mb={1} w="full">
    <FormLabel m={0} minW="90px">
      {label}
    </FormLabel>
    <CompositeSlider value={value} onChange={onChange} defaultValue={0} min={min} max={max} step={step} marks />
    <CompositeNumberInput value={value} onChange={onChange} defaultValue={0} min={min} max={max} step={step} />
  </FormControl>
);

const DEFAULT_SIMPLE_ADJUSTMENTS = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  tint: 0,
  sharpness: 0,
};

export const RasterLayerSimpleAdjustmentsEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const selectSimpleAdjustments = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.simple ?? DEFAULT_SIMPLE_ADJUSTMENTS
    );
  }, [entityIdentifier]);
  const simple = useAppSelector(selectSimpleAdjustments);
  const selectIsDisabled = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.enabled !== true
    );
  }, [entityIdentifier]);
  const isDisabled = useAppSelector(selectIsDisabled);

  const onBrightness = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { brightness: v } })),
    [dispatch, entityIdentifier]
  );
  const onContrast = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { contrast: v } })),
    [dispatch, entityIdentifier]
  );
  const onSaturation = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { saturation: v } })),
    [dispatch, entityIdentifier]
  );
  const onTemperature = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { temperature: v } })),
    [dispatch, entityIdentifier]
  );
  const onTint = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { tint: v } })),
    [dispatch, entityIdentifier]
  );
  const onSharpness = useCallback(
    (v: number) => dispatch(rasterLayerAdjustmentsSimpleUpdated({ entityIdentifier, simple: { sharpness: v } })),
    [dispatch, entityIdentifier]
  );

  return (
    <Flex px={3} pb={2} direction="column" opacity={isDisabled ? 0.3 : 1} pointerEvents={isDisabled ? 'none' : 'auto'}>
      <AdjustmentSliderRow
        label={t('controlLayers.adjustments.brightness')}
        value={simple.brightness}
        onChange={onBrightness}
      />
      <AdjustmentSliderRow
        label={t('controlLayers.adjustments.contrast')}
        value={simple.contrast}
        onChange={onContrast}
      />
      <AdjustmentSliderRow
        label={t('controlLayers.adjustments.saturation')}
        value={simple.saturation}
        onChange={onSaturation}
      />
      <AdjustmentSliderRow
        label={t('controlLayers.adjustments.temperature')}
        value={simple.temperature}
        onChange={onTemperature}
      />
      <AdjustmentSliderRow label={t('controlLayers.adjustments.tint')} value={simple.tint} onChange={onTint} />
      <AdjustmentSliderRow
        label={t('controlLayers.adjustments.sharpness')}
        value={simple.sharpness}
        onChange={onSharpness}
        min={-0.5}
        max={0.5}
        step={0.01}
      />
    </Flex>
  );
});

RasterLayerSimpleAdjustmentsEditor.displayName = 'RasterLayerSimpleAdjustmentsEditor';
