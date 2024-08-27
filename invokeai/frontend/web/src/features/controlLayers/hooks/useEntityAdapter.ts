import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityLayerAdapter } from 'features/controlLayers/konva/CanvasEntityLayerAdapter';
import type { CanvasEntityMaskAdapter } from 'features/controlLayers/konva/CanvasEntityMaskAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useEntityAdapter = (
  entityIdentifier: CanvasEntityIdentifier
): CanvasEntityLayerAdapter | CanvasEntityMaskAdapter => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    const entity = canvasManager.stateApi.getEntity(entityIdentifier);
    assert(entity, 'Entity adapter not found');
    return entity.adapter;
  }, [canvasManager.stateApi, entityIdentifier]);

  return adapter;
};
