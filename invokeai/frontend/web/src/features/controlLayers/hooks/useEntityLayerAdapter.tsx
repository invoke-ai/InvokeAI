import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

const EntityLayerAdapterContext = createContext<CanvasLayerAdapter | null>(null);

export const EntityLayerAdapterProviderGate = memo(({ children }: PropsWithChildren) => {
  const entityIdentifier = useEntityIdentifierContext();
  const canvasManager = useCanvasManager();
  const adapter = useMemo(() => {
    if (entityIdentifier.type === 'raster_layer') {
      return canvasManager.rasterLayerAdapters.get(entityIdentifier.id) ?? null;
    } else if (entityIdentifier.type === 'control_layer') {
      return canvasManager.controlLayerAdapters.get(entityIdentifier.id) ?? null;
    }
    assert(false, 'EntityLayerAdapterProviderGate must be used with a valid EntityIdentifierContext');
  }, [canvasManager, entityIdentifier]);

  if (!canvasManager) {
    return null;
  }

  return <EntityLayerAdapterContext.Provider value={adapter}>{children}</EntityLayerAdapterContext.Provider>;
});

EntityLayerAdapterProviderGate.displayName = 'EntityLayerAdapterProviderGate';

export const useEntityLayerAdapter = (): CanvasLayerAdapter => {
  const adapter = useContext(EntityLayerAdapterContext);

  assert(adapter, 'useEntityLayerAdapter must be used within a EntityLayerAdapterProviderGate');

  return adapter;
};
