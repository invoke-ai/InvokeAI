import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReadableAtom } from 'nanostores';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';

const $fallbackFalse: ReadableAtom<boolean> = atom(false);

export const useEntityTransform = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isInteractable = useStore(adapter?.$isInteractable ?? $fallbackFalse);

  const start = useCallback(() => {
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
    adapter.transformer.startTransform();
  }, [entityIdentifier, canvasManager]);

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
    return false;
  }, [entityIdentifier, adapter, isBusy, isInteractable]);

  return { isDisabled, start } as const;
};
