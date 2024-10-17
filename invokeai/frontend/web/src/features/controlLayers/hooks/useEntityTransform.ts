import { useStore } from '@nanostores/react';
import { $false } from 'app/store/nanostores/util';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import { useCallback, useMemo } from 'react';

export const useEntityTransform = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isInteractable = useStore(adapter?.$isInteractable ?? $false);
  const isEmpty = useStore(adapter?.$isEmpty ?? $false);

  const isDisabled = useMemo(() => {
    if (!entityIdentifier) {
      return true;
    }
    if (!isTransformableEntityIdentifier(entityIdentifier)) {
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

  const start = useCallback(async () => {
    if (isDisabled) {
      return;
    }
    if (!entityIdentifier) {
      return;
    }
    if (!isTransformableEntityIdentifier(entityIdentifier)) {
      return;
    }
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter) {
      return;
    }
    await adapter.transformer.startTransform();
  }, [isDisabled, entityIdentifier, canvasManager]);

  const fitToBbox = useCallback(async () => {
    if (isDisabled) {
      return;
    }
    if (!entityIdentifier) {
      return;
    }
    if (!isTransformableEntityIdentifier(entityIdentifier)) {
      return;
    }
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter) {
      return;
    }
    await adapter.transformer.startTransform({ silent: true });
    adapter.transformer.fitToBboxContain();
    await adapter.transformer.applyTransform();
  }, [canvasManager, entityIdentifier, isDisabled]);

  return { isDisabled, start, fitToBbox } as const;
};
