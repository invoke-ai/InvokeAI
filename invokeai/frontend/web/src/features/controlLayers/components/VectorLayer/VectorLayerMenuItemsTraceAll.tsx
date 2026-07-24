import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasBrushLineState, CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { approximateBezierPath } from 'features/controlLayers/util/bezierPath';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWaveSineBold } from 'react-icons/pi';

export const VectorLayerMenuItemsTraceAll = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const onClick = useCallback(() => {
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    const { paths, position } = adapter.state;

    const editSession = canvasManager.tool.tools.path.$editSession.get();
    if (
      editSession &&
      editSession.entityIdentifier.id === adapter.entityIdentifier.id &&
      editSession.entityIdentifier.type === adapter.entityIdentifier.type
    ) {
      canvasManager.tool.tools.path.acceptEditSession();
    }

    const color = canvasManager.stateApi.getCurrentColor();
    const brushWidth = canvasManager.stateApi.getSettings().brushWidth;

    const objects: CanvasBrushLineState[] = paths.flatMap((path) => {
      const coordinates = approximateBezierPath(path.points, path.isClosed);
      if (coordinates.length < 2) {
        return [];
      }

      return [
        {
          id: getPrefixedId('brush_line'),
          type: 'brush_line',
          strokeWidth: brushWidth,
          points: coordinates.flatMap((coordinate) => [coordinate.x, coordinate.y]),
          color,
          clip: null,
        },
      ];
    });

    if (objects.length === 0) {
      return;
    }

    canvasManager.stateApi.addRasterLayer({
      isSelected: true,
      overrides: {
        position: { ...position },
        objects,
      } satisfies Partial<CanvasRasterLayerState>,
    });
  }, [adapter, canvasManager]);

  if (!adapter || adapter.state.type !== 'vector_layer') {
    return null;
  }

  const { paths } = adapter.state;

  return (
    <MenuItem onClick={onClick} icon={<PiWaveSineBold />} isDisabled={paths.length === 0}>
      {t('controlLayers.vectorEdit.traceAll')}
    </MenuItem>
  );
});

VectorLayerMenuItemsTraceAll.displayName = 'VectorLayerMenuItemsTraceAll';
