import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasControlLayerAdapter } from 'features/controlLayers/konva/CanvasControlLayerAdapter';
import type { CanvasInpaintMaskAdapter } from 'features/controlLayers/konva/CanvasInpaintMaskAdapter';
import type { CanvasRasterLayerAdapter } from 'features/controlLayers/konva/CanvasRasterLayerAdapter';
import type { CanvasRegionalGuidanceAdapter } from 'features/controlLayers/konva/CanvasRegionalGuidanceAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useEntityAdapter = (
  entityIdentifier: CanvasEntityIdentifier
): CanvasRasterLayerAdapter | CanvasControlLayerAdapter | CanvasInpaintMaskAdapter | CanvasRegionalGuidanceAdapter => {
  const canvasManager = useCanvasManager();

  const adapter = useMemo(() => {
    const entity = canvasManager.stateApi.getEntity(entityIdentifier);
    assert(entity, 'Entity adapter not found');
    return entity.adapter;
  }, [canvasManager.stateApi, entityIdentifier]);

  return adapter;
};
