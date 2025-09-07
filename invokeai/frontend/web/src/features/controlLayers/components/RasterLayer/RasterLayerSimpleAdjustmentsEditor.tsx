import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsSimpleUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback } from 'react';
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
  <FormControl pr={2}>
    <Flex alignItems="center" gap={3} mb={1}>
      <FormLabel m={0} flexShrink={0} minW="90px">
        {label}
      </FormLabel>
      <CompositeNumberInput value={value} onChange={onChange} min={min} max={max} step={step} flex="0 0 96px" />
    </Flex>
    <CompositeSlider value={value} onChange={onChange} min={min} max={max} step={step} marks />
  </FormControl>
);

export const RasterLayerSimpleAdjustmentsEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const layer = useAppSelector((s) => selectEntity(s.canvas.present, entityIdentifier));

  const simple = layer?.adjustments?.simple ?? {
    brightness: 0,
    contrast: 0,
    saturation: 0,
    temperature: 0,
    tint: 0,
    sharpness: 0,
  };

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
    <>
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
        min={0}
        max={1}
        step={0.01}
      />
    </>
  );
});

RasterLayerSimpleAdjustmentsEditor.displayName = 'RasterLayerSimpleAdjustmentsEditor';
