import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsEnabled } from 'features/controlLayers/hooks/useEntityIsEnabled';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import type { CanvasObjectState, CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { buildVectorTraceObject } from 'features/controlLayers/util/vectorLayerTrace';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWaveSineBold } from 'react-icons/pi';

export const VectorLayerMenuItemsTraceAll = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isEnabled = useEntityIsEnabled(entityIdentifier);
  const isLocked = useEntityIsLocked(entityIdentifier);
  const isVectorLayerTypeHidden = useEntityTypeIsHidden('vector_layer');
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
    const { brushWidth, traceTaperEnds, traceTaper } = canvasManager.stateApi.getSettings();

    const objects: CanvasObjectState[] = paths.flatMap((path): CanvasObjectState[] => {
      const object = buildVectorTraceObject(path, brushWidth, color, traceTaperEnds, traceTaper);
      return object ? [object] : [];
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
    <MenuItem
      onClick={onClick}
      icon={<PiWaveSineBold />}
      isDisabled={paths.length === 0 || !isEnabled || isLocked || isVectorLayerTypeHidden || isBusy}
    >
      {t('controlLayers.vectorEdit.traceAll')}
    </MenuItem>
  );
});

VectorLayerMenuItemsTraceAll.displayName = 'VectorLayerMenuItemsTraceAll';
