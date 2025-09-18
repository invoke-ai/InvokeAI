import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsEmpty } from 'features/controlLayers/hooks/useEntityIsEmpty';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isRasterLayerEntityIdentifier } from 'features/controlLayers/store/types';
import { useCallback, useMemo } from 'react';

export const useEntityWorkflowTrigger = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);
  const isEmpty = useEntityIsEmpty(entityIdentifier);
  const workflowTrigger = useStore(canvasManager.stateApi.$workflowTrigger);

  const isDisabled = useMemo(() => {
    if (!entityIdentifier) {
      return true;
    }
    if (!isRasterLayerEntityIdentifier(entityIdentifier)) {
      return true;
    }
    if (!adapter) {
      return true;
    }
    if (!(adapter instanceof CanvasEntityAdapterRasterLayer)) {
      return true;
    }
    if (isBusy) {
      return true;
    }
    if (isLocked) {
      return true;
    }
    if (isEmpty) {
      return true;
    }
    if (workflowTrigger && workflowTrigger.adapter !== adapter) {
      return true;
    }
    return false;
  }, [adapter, entityIdentifier, isBusy, isEmpty, isLocked, workflowTrigger]);

  const start = useCallback(() => {
    if (isDisabled) {
      return;
    }
    if (!entityIdentifier || !isRasterLayerEntityIdentifier(entityIdentifier) || !adapter) {
      return;
    }
    if (!(adapter instanceof CanvasEntityAdapterRasterLayer)) {
      return;
    }
    canvasManager.stateApi.startWorkflowTrigger(adapter);
  }, [adapter, canvasManager.stateApi, entityIdentifier, isDisabled]);

  return { isDisabled, start } as const;
};
