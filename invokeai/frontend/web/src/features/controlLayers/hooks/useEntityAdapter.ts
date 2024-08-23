import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useEntityAdapter = (entityIdentifier: CanvasEntityIdentifier): CanvasLayerAdapter | CanvasMaskAdapter => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    const entity = canvasManager.stateApi.getEntity(entityIdentifier);
    assert(entity, 'Entity adapter not found');
    return entity.adapter;
  }, [canvasManager.stateApi, entityIdentifier]);

  return adapter;
};
