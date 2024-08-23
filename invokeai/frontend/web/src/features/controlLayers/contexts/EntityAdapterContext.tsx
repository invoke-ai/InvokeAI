import type { SyncableMap } from 'common/util/SyncableMap/SyncableMap';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo, useSyncExternalStore } from 'react';
import { assert } from 'tsafe';

const EntityAdapterContext = createContext<CanvasLayerAdapter | CanvasMaskAdapter | null>(null);

export const EntityLayerAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const store = useMemo<SyncableMap<string, CanvasLayerAdapter>>(() => {
    if (entityIdentifier.type === 'raster_layer') {
      return canvasManager.adapters.rasterLayers;
    }
    if (entityIdentifier.type === 'control_layer') {
      return canvasManager.adapters.controlLayers;
    }
    assert(false, 'Unknown entity type');
  }, [canvasManager.adapters.controlLayers, canvasManager.adapters.rasterLayers, entityIdentifier.type]);
  const adapters = useSyncExternalStore(store.subscribe, store.getSnapshot);
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

EntityLayerAdapterGate.displayName = 'EntityLayerAdapterGate';

/** @knipignore */
export const useEntityLayerAdapter = (): CanvasLayerAdapter => {
  const adapter = useContext(EntityAdapterContext);
  assert(adapter, 'useEntityLayerAdapter must be used within a EntityLayerAdapterGate');
  assert(adapter.type === 'layer_adapter', 'useEntityLayerAdapter must be used with a layer adapter');
  return adapter;
};

export const EntityMaskAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const store = useMemo<SyncableMap<string, CanvasMaskAdapter>>(() => {
    if (entityIdentifier.type === 'inpaint_mask') {
      return canvasManager.adapters.inpaintMasks;
    }
    if (entityIdentifier.type === 'regional_guidance') {
      return canvasManager.adapters.regionMasks;
    }
    assert(false, 'Unknown entity type');
  }, [canvasManager.adapters.inpaintMasks, canvasManager.adapters.regionMasks, entityIdentifier.type]);
  const adapters = useSyncExternalStore(store.subscribe, store.getSnapshot);
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

EntityMaskAdapterGate.displayName = 'EntityMaskAdapterGate';

/** @knipignore */
export const useEntityMaskAdapter = (): CanvasMaskAdapter => {
  const adapter = useContext(EntityAdapterContext);
  assert(adapter, 'useEntityMaskAdapter must be used within a CanvasMaskAdapterGate');
  assert(adapter.type === 'mask_adapter', 'useEntityMaskAdapter must be used with a mask adapter');
  return adapter;
};

export const useEntityAdapter = (): CanvasLayerAdapter | CanvasMaskAdapter => {
  const adapter = useContext(EntityAdapterContext);
  assert(adapter, 'useEntityAdapter must be used within a CanvasRasterLayerAdapterGate');
  return adapter;
};
