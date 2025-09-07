import { Button, ButtonGroup, Flex, IconButton, Switch, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RasterLayerCurvesAdjustmentsEditor } from 'features/controlLayers/components/RasterLayer/RasterLayerCurvesAdjustmentsEditor';
import { RasterLayerSimpleAdjustmentsEditor } from 'features/controlLayers/components/RasterLayer/RasterLayerSimpleAdjustmentsEditor';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rasterLayerAdjustmentsCancel,
  rasterLayerAdjustmentsCurvesUpdated,
  rasterLayerAdjustmentsSet,
  rasterLayerAdjustmentsSimpleUpdated,
} from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import { makeDefaultRasterLayerAdjustments } from 'features/controlLayers/store/util';
import React, { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCaretDownBold, PiCheckBold, PiTrashBold } from 'react-icons/pi';

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

  const onToggleEnabled = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = e.target.checked;
      const current = layer?.adjustments ?? makeDefaultRasterLayerAdjustments(mode);
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: { ...current, enabled: v },
        })
      );
    },
    [dispatch, entityIdentifier, layer?.adjustments, mode]
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

  const onCancel = useCallback(() => {
    // Clear out adjustments entirely
    dispatch(rasterLayerAdjustmentsCancel({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const onToggleCollapsed = useCallback(() => {
    const current = layer?.adjustments ?? makeDefaultRasterLayerAdjustments(mode);
    dispatch(
      rasterLayerAdjustmentsSet({
        entityIdentifier,
        adjustments: { ...current, collapsed: !collapsed },
      })
    );
  }, [dispatch, entityIdentifier, collapsed, layer?.adjustments, mode]);

  const onSetMode = useCallback(
    (nextMode: 'simple' | 'curves') => {
      if (nextMode === mode) {
        return;
      }
      const current = layer?.adjustments ?? makeDefaultRasterLayerAdjustments(nextMode);
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: { ...current, mode: nextMode },
        })
      );
    },
    [dispatch, entityIdentifier, layer?.adjustments, mode]
  );

  // Memoized click handlers to avoid inline arrow functions in JSX
  const onClickModeSimple = useCallback(() => onSetMode('simple'), [onSetMode]);
  const onClickModeCurves = useCallback(() => onSetMode('curves'), [onSetMode]);

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
      <Flex px={4} alignItems="center" gap={3} mt={2} mb={2}>
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
          <Button
            onClick={onClickModeSimple}
            isActive={mode === 'simple'}
            colorScheme={mode === 'simple' ? 'invokeBlue' : undefined}
          >
            {t('controlLayers.adjustments.simple')}
          </Button>
          <Button
            onClick={onClickModeCurves}
            isActive={mode === 'curves'}
            colorScheme={mode === 'curves' ? 'invokeBlue' : undefined}
          >
            {t('controlLayers.adjustments.curves')}
          </Button>
        </ButtonGroup>
        <Switch isChecked={enabled} onChange={onToggleEnabled} />
        <IconButton
          aria-label={t('controlLayers.adjustments.cancel')}
          size="md"
          onClick={onCancel}
          isDisabled={!layer?.adjustments}
          colorScheme="red"
          icon={<PiTrashBold />}
          variant="ghost"
        />
        <IconButton
          aria-label={t('controlLayers.adjustments.reset')}
          size="md"
          onClick={onReset}
          isDisabled={!layer?.adjustments}
          icon={<PiArrowCounterClockwiseBold />}
          variant="ghost"
        />
        <IconButton
          aria-label={t('controlLayers.adjustments.finish')}
          size="md"
          onClick={onFinish}
          isDisabled={!layer?.adjustments}
          colorScheme="green"
          icon={<PiCheckBold />}
          variant="ghost"
        />
      </Flex>

      {!collapsed && mode === 'simple' && <RasterLayerSimpleAdjustmentsEditor />}

      {!collapsed && mode === 'curves' && <RasterLayerCurvesAdjustmentsEditor />}
    </>
  );
});

RasterLayerAdjustmentsPanel.displayName = 'RasterLayerAdjustmentsPanel';
