import { Button, ButtonGroup, Flex, IconButton, Switch, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RasterLayerCurvesAdjustmentsEditor } from 'features/controlLayers/components/RasterLayer/RasterLayerCurvesAdjustmentsEditor';
import { RasterLayerSimpleAdjustmentsEditor } from 'features/controlLayers/components/RasterLayer/RasterLayerSimpleAdjustmentsEditor';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rasterLayerAdjustmentsCancel,
  rasterLayerAdjustmentsCollapsedToggled,
  rasterLayerAdjustmentsEnabledToggled,
  rasterLayerAdjustmentsModeChanged,
  rasterLayerAdjustmentsReset,
  rasterLayerAdjustmentsSet,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCaretDownBold, PiCheckBold, PiTrashBold } from 'react-icons/pi';

export const RasterLayerAdjustmentsPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const canvasManager = useCanvasManager();

  const selectHasAdjustments = useMemo(() => {
    return createSelector(selectCanvasSlice, (canvas) => Boolean(selectEntity(canvas, entityIdentifier)?.adjustments));
  }, [entityIdentifier]);

  const hasAdjustments = useAppSelector(selectHasAdjustments);

  const selectMode = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.mode ?? 'simple'
    );
  }, [entityIdentifier]);
  const mode = useAppSelector(selectMode);

  const selectEnabled = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.enabled ?? false
    );
  }, [entityIdentifier]);
  const enabled = useAppSelector(selectEnabled);

  const selectCollapsed = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.collapsed ?? false
    );
  }, [entityIdentifier]);
  const collapsed = useAppSelector(selectCollapsed);

  const onToggleEnabled = useCallback(() => {
    dispatch(rasterLayerAdjustmentsEnabledToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const onReset = useCallback(() => {
    // Reset values to defaults but keep adjustments present; preserve enabled/collapsed/mode
    dispatch(rasterLayerAdjustmentsReset({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const onCancel = useCallback(() => {
    // Clear out adjustments entirely
    dispatch(rasterLayerAdjustmentsCancel({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const onToggleCollapsed = useCallback(() => {
    dispatch(rasterLayerAdjustmentsCollapsedToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const onClickModeSimple = useCallback(
    () => dispatch(rasterLayerAdjustmentsModeChanged({ entityIdentifier, mode: 'simple' })),
    [dispatch, entityIdentifier]
  );

  const onClickModeCurves = useCallback(
    () => dispatch(rasterLayerAdjustmentsModeChanged({ entityIdentifier, mode: 'curves' })),
    [dispatch, entityIdentifier]
  );

  const onFinish = useCallback(async () => {
    // Bake current visual into layer pixels, then clear adjustments
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter || adapter.type !== 'raster_layer_adapter') {
      return;
    }
    const rect = adapter.transformer.getRelativeRect();
    try {
      await adapter.renderer.rasterize({ rect, replaceObjects: true, attrs: { opacity: 1 } });
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
      <Flex px={2} pb={2} alignItems="center" gap={2}>
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
          <Button onClick={onClickModeSimple} colorScheme={mode === 'simple' ? 'invokeBlue' : undefined}>
            {t('controlLayers.adjustments.simple')}
          </Button>
          <Button onClick={onClickModeCurves} colorScheme={mode === 'curves' ? 'invokeBlue' : undefined}>
            {t('controlLayers.adjustments.curves')}
          </Button>
        </ButtonGroup>
        <Switch isChecked={enabled} onChange={onToggleEnabled} />
        <IconButton
          aria-label={t('controlLayers.adjustments.cancel')}
          size="md"
          onClick={onCancel}
          isDisabled={!hasAdjustments}
          colorScheme="red"
          icon={<PiTrashBold />}
          variant="ghost"
        />
        <IconButton
          aria-label={t('controlLayers.adjustments.reset')}
          size="md"
          onClick={onReset}
          isDisabled={!hasAdjustments}
          icon={<PiArrowCounterClockwiseBold />}
          variant="ghost"
        />
        <IconButton
          aria-label={t('controlLayers.adjustments.finish')}
          size="md"
          onClick={onFinish}
          isDisabled={!hasAdjustments}
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
