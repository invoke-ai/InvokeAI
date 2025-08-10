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
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rasterLayerAdjustmentsCurvesUpdated,
  rasterLayerAdjustmentsSet,
  rasterLayerAdjustmentsSimpleUpdated,
} from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const RasterLayerAdjustmentsPanel = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
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
      dispatch(
        rasterLayerAdjustmentsSet({ entityIdentifier, adjustments: { enabled: v, collapsed: false, mode: 'simple' } })
      );
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

  const slider = useMemo(
    () =>
      ({
        row: (label: string, value: number, onChange: (v: number) => void, min = -1, max = 1, step = 0.01) => (
          <FormControl pr={2}>
            <Flex alignItems="center" gap={3} mb={1}>
              <FormLabel m={0} flexShrink={0} minW="90px">
                {label}
              </FormLabel>
              <CompositeNumberInput value={value} onChange={onChange} min={min} max={max} step={step} flex="0 0 96px" />
            </Flex>
            <CompositeSlider value={value} onChange={onChange} min={min} max={max} step={step} marks />
          </FormControl>
        ),
      }) as const,
    []
  );

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
            {t('controlLayers.adjustments.simple', { defaultValue: 'Simple' })}
          </Button>
          <Button onClick={onClickModeCurves} isActive={mode === 'curves'}>
            {t('controlLayers.adjustments.curves', { defaultValue: 'Curves' })}
          </Button>
        </ButtonGroup>
        <Switch isChecked={enabled} onChange={handleToggleEnabled} />
        <Button size="sm" onClick={onReset} isDisabled={!layer?.adjustments}>
          Reset
        </Button>
      </Flex>

      {!collapsed && mode === 'simple' && (
        <>
          {slider.row(t('controlLayers.adjustments.brightness'), simple.brightness, onBrightness)}
          {slider.row(t('controlLayers.adjustments.contrast'), simple.contrast, onContrast)}
          {slider.row(t('controlLayers.adjustments.saturation'), simple.saturation, onSaturation)}
          {slider.row(t('controlLayers.adjustments.temperature'), simple.temperature, onTemperature)}
          {slider.row(t('controlLayers.adjustments.tint'), simple.tint, onTint)}
          {slider.row(t('controlLayers.adjustments.sharpness'), simple.sharpness, onSharpness, 0, 1, 0.01)}
        </>
      )}

      {!collapsed && mode === 'curves' && <RasterLayerCurvesEditor />}
    </>
  );
});

RasterLayerAdjustmentsPanel.displayName = 'RasterLayerAdjustmentsPanel';
