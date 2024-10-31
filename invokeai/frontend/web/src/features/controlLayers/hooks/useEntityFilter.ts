import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsEmpty } from 'features/controlLayers/hooks/useEntityIsEmpty';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useCallback, useMemo } from 'react';

export const useEntityFilter = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);
  const isEmpty = useEntityIsEmpty(entityIdentifier);

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
    if (isLocked) {
      return true;
    }
    if (isEmpty) {
      return true;
    }
    return false;
  }, [entityIdentifier, adapter, isBusy, isLocked, isEmpty]);

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
    imageViewer.close();
    adapter.filterer.start();
  }, [isDisabled, entityIdentifier, canvasManager, imageViewer]);

  return { isDisabled, start } as const;
};
