import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReadableAtom } from 'nanostores';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';

const $fallbackFalse: ReadableAtom<boolean> = atom(false);

export const useEntityFilter = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isInteractable = useStore(adapter?.$isInteractable ?? $fallbackFalse);

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
    return false;
  }, [entityIdentifier, adapter, isBusy, isInteractable]);

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
