import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntityAdapter/types';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

/** @knipignore */
export const useEntityAdapter = (entityIdentifier: CanvasEntityIdentifier): CanvasEntityAdapter => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    const adapter = canvasManager.getAdapter(entityIdentifier);
    assert(adapter, 'Entity adapter not found');
    return adapter;
  }, [canvasManager, entityIdentifier]);

  return adapter;
};

export const useEntityAdapterSafe = (entityIdentifier: CanvasEntityIdentifier | null): CanvasEntityAdapter | null => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    if (!entityIdentifier) {
      return null;
    }
    const adapter = canvasManager.getAdapter(entityIdentifier);
    return adapter ?? null;
  }, [canvasManager, entityIdentifier]);

  return adapter;
};
