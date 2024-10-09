import { useStore } from '@nanostores/react';
import { $false } from 'app/store/nanostores/util';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import { useCallback, useMemo } from 'react';

export const useEntityFilter = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isInteractable = useStore(adapter?.$isInteractable ?? $false);
  const isEmpty = useStore(adapter?.$isEmpty ?? $false);

  const isDisabled = useMemo(() => {
    if (!entityIdentifier) {
      return true;
    }
    if (!isFilterableEntityIdentifier(entityIdentifier)) {
      return true;
    }
    if (!adapter) {
      return true;
    }
    if (isBusy) {
      return true;
    }
    if (!isInteractable) {
      return true;
    }
    if (isEmpty) {
      return true;
    }
    return false;
  }, [entityIdentifier, adapter, isBusy, isInteractable, isEmpty]);

  const start = useCallback(() => {
    if (isDisabled) {
      return;
    }
    if (!entityIdentifier) {
      return;
    }
    if (!isFilterableEntityIdentifier(entityIdentifier)) {
      return;
    }
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter) {
      return;
    }
    adapter.filterer.start();
  }, [isDisabled, entityIdentifier, canvasManager]);

  return { isDisabled, start } as const;
};
