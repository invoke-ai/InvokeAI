import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasInpaintMaskState, CanvasRasterLayerState } from 'features/controlLayers/store/types';
import {
  buildClosedPathLassoObjects,
  buildClosedPathPolygonObjects,
  isFillableBezierPath,
} from 'features/controlLayers/util/vectorLayerMaterialization';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPaintBucketBold, PiSelectionAllBold } from 'react-icons/pi';

export const VectorLayerMenuItemsMaterializeClosedPaths = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);

  const acceptActiveEditSession = useCallback(() => {
    const editSession = canvasManager.tool.tools.path.$editSession.get();
    if (
      editSession &&
      editSession.entityIdentifier.id === entityIdentifier.id &&
      editSession.entityIdentifier.type === entityIdentifier.type
    ) {
      canvasManager.tool.tools.path.acceptEditSession();
    }
  }, [canvasManager, entityIdentifier]);

  const onFillClosedPaths = useCallback(() => {
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    acceptActiveEditSession();
    const objects = buildClosedPathPolygonObjects(adapter.state.paths, canvasManager.stateApi.getCurrentColor());
    if (objects.length === 0) {
      return;
    }

    canvasManager.stateApi.addRasterLayer({
      isSelected: true,
      overrides: {
        position: { ...adapter.state.position },
        objects,
      } satisfies Partial<CanvasRasterLayerState>,
    });
  }, [acceptActiveEditSession, adapter, canvasManager]);

  const onCreateInpaintMask = useCallback(() => {
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    acceptActiveEditSession();
    const objects = buildClosedPathLassoObjects(adapter.state.paths);
    if (objects.length === 0) {
      return;
    }

    canvasManager.stateApi.addInpaintMask({
      isSelected: true,
      overrides: {
        position: { ...adapter.state.position },
        objects,
      } satisfies Partial<CanvasInpaintMaskState>,
    });
  }, [acceptActiveEditSession, adapter, canvasManager]);

  if (!adapter || adapter.state.type !== 'vector_layer') {
    return null;
  }

  const hasClosedPaths = adapter.state.paths.some(isFillableBezierPath);

  return (
    <>
      <MenuItem onClick={onFillClosedPaths} icon={<PiPaintBucketBold />} isDisabled={!hasClosedPaths}>
        {t('controlLayers.vectorEdit.fillClosedPaths')}
      </MenuItem>
      <MenuItem onClick={onCreateInpaintMask} icon={<PiSelectionAllBold />} isDisabled={!hasClosedPaths}>
        {t('controlLayers.vectorEdit.createInpaintMask')}
      </MenuItem>
    </>
  );
});

VectorLayerMenuItemsMaterializeClosedPaths.displayName = 'VectorLayerMenuItemsMaterializeClosedPaths';
