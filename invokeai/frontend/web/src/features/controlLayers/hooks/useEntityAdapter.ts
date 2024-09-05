import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntityAdapterRegionalGuidance';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useEntityAdapter = (
  entityIdentifier: CanvasEntityIdentifier
): CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer | CanvasEntityAdapterInpaintMask | CanvasEntityAdapterRegionalGuidance => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    const entity = canvasManager.stateApi.getEntity(entityIdentifier);
    assert(entity, 'Entity adapter not found');
    return entity.adapter;
  }, [canvasManager.stateApi, entityIdentifier]);

  return adapter;
};
