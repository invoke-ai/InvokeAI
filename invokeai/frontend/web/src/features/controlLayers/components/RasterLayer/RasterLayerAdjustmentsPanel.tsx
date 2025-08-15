import {
  Button,
  ButtonGroup,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Switch,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RasterLayerCurvesEditor } from 'features/controlLayers/components/RasterLayer/RasterLayerCurvesEditor';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rasterLayerAdjustmentsCurvesUpdated,
  rasterLayerAdjustmentsSet,
  rasterLayerAdjustmentsSimpleUpdated,
} from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

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

export const RasterLayerAdjustmentsPanel = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const canvasManager = useCanvasManager();
  const layer = useAppSelector((s) => selectEntity(s.canvas.present, entityIdentifier));
  const { t } = useTranslation();

  const hasAdjustments = Boolean(layer?.adjustments);
  const enabled = Boolean(layer?.adjustments?.enabled);
  const collapsed = Boolean(layer?.adjustments?.collapsed);
  const mode = layer?.adjustments?.mode ?? 'simple';
  const simple = layer?.adjustments?.simple ?? {
    brightness: 0,
    contrast: 0,
    saturation: 0,
    temperature: 0,
    tint: 0,
    sharpness: 0,
  };

  const onToggleEnabled = useCallback(
    (v: boolean) => {
      // Only toggle the enabled state; preserve current mode/collapsed so users can A/B compare
      dispatch(rasterLayerAdjustmentsSet({ entityIdentifier, adjustments: { enabled: v } }));
    },
    [dispatch, entityIdentifier]
  );

  const onReset = useCallback(() => {
    // Reset values to defaults but keep adjustments present; preserve enabled/collapsed/mode
    dispatch(
      rasterLayerAdjustmentsSimpleUpdated({
        entityIdentifier,
        simple: {
          brightness: 0,
          contrast: 0,
          saturation: 0,
          temperature: 0,
          tint: 0,
          sharpness: 0,
        },
      })
    );
    const defaultPoints: Array<[number, number]> = [
      [0, 0],
      [255, 255],
    ];
    dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel: 'master', points: defaultPoints }));
    dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel: 'r', points: defaultPoints }));
    dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel: 'g', points: defaultPoints }));
    dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel: 'b', points: defaultPoints }));
  }, [dispatch, entityIdentifier]);

  const onToggleCollapsed = useCallback(() => {
    dispatch(
      rasterLayerAdjustmentsSet({
        entityIdentifier,
        adjustments: { collapsed: !collapsed },
      })
    );
  }, [dispatch, entityIdentifier, collapsed]);

  const onSetMode = useCallback(
    (nextMode: 'simple' | 'curves') => {
      if (!layer?.adjustments) {
        return;
      }
      if (nextMode === mode) {
        return;
      }
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: { mode: nextMode },
        })
      );
    },
    [dispatch, entityIdentifier, layer?.adjustments, mode]
  );

  // Memoized click handlers to avoid inline arrow functions in JSX
  const onClickModeSimple = useCallback(() => onSetMode('simple'), [onSetMode]);
  const onClickModeCurves = useCallback(() => onSetMode('curves'), [onSetMode]);

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

  const handleToggleEnabled = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => onToggleEnabled(e.target.checked),
    [onToggleEnabled]
  );

  const onFinish = useCallback(async () => {
    // Bake current visual into layer pixels, then clear adjustments
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter || adapter.type !== 'raster_layer_adapter') {
      return;
    }
    const rect = adapter.transformer.getRelativeRect();
    try {
      await adapter.renderer.rasterize({ rect, replaceObjects: true });
      // Clear adjustments after baking
      dispatch(rasterLayerAdjustmentsSet({ entityIdentifier, adjustments: null }));
    } catch {
      // no-op; leave state unchanged on failure
    }
  }, [canvasManager, entityIdentifier, dispatch]);

  // Hide the panel entirely until adjustments are added via context menu
  if (!hasAdjustments) {
    return null;
  }

  return (
    <>
      <Flex alignItems="center" gap={3} mt={2} mb={2}>
        <IconButton
          aria-label={collapsed ? t('controlLayers.adjustments.expand') : t('controlLayers.adjustments.collapse')}
          size="sm"
          variant="ghost"
          onClick={onToggleCollapsed}
          icon={
            <PiCaretDownBold
              style={{ transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
            />
          }
        />
        <Text fontWeight={600} flex={1}>
          Adjustments
        </Text>
        <ButtonGroup size="sm" isAttached variant="outline">
          <Button onClick={onClickModeSimple} isActive={mode === 'simple'}>
            {t('controlLayers.adjustments.simple')}
          </Button>
          <Button onClick={onClickModeCurves} isActive={mode === 'curves'}>
            {t('controlLayers.adjustments.curves')}
          </Button>
        </ButtonGroup>
        <Switch isChecked={enabled} onChange={handleToggleEnabled} />
        <Button size="sm" onClick={onReset} isDisabled={!layer?.adjustments} colorScheme="red">
          {t('controlLayers.adjustments.reset')}
        </Button>
        <Button size="sm" onClick={onFinish} isDisabled={!layer?.adjustments} colorScheme="green">
          {t('controlLayers.adjustments.finish')}
        </Button>
      </Flex>

      {!collapsed && mode === 'simple' && (
        <>
          <AdjustmentSliderRow label={t('controlLayers.adjustments.brightness')} value={simple.brightness} onChange={onBrightness} />
          <AdjustmentSliderRow label={t('controlLayers.adjustments.contrast')} value={simple.contrast} onChange={onContrast} />
          <AdjustmentSliderRow label={t('controlLayers.adjustments.saturation')} value={simple.saturation} onChange={onSaturation} />
          <AdjustmentSliderRow label={t('controlLayers.adjustments.temperature')} value={simple.temperature} onChange={onTemperature} />
          <AdjustmentSliderRow label={t('controlLayers.adjustments.tint')} value={simple.tint} onChange={onTint} />
          <AdjustmentSliderRow label={t('controlLayers.adjustments.sharpness')} value={simple.sharpness} onChange={onSharpness} min={0} max={1} step={0.01} />
        </>
      )}

      {!collapsed && mode === 'curves' && <RasterLayerCurvesEditor />}
    </>
  );
});

RasterLayerAdjustmentsPanel.displayName = 'RasterLayerAdjustmentsPanel';
