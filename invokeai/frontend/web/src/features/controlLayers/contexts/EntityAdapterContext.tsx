import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasControlLayerAdapter } from 'features/controlLayers/konva/CanvasControlLayerAdapter';
import type { CanvasInpaintMaskAdapter } from 'features/controlLayers/konva/CanvasInpaintMaskAdapter';
import type { CanvasRasterLayerAdapter } from 'features/controlLayers/konva/CanvasRasterLayerAdapter';
import type { CanvasRegionalGuidanceAdapter } from 'features/controlLayers/konva/CanvasRegionalGuidanceAdapter';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo, useSyncExternalStore } from 'react';
import { assert } from 'tsafe';

const EntityAdapterContext = createContext<
  CanvasRasterLayerAdapter | CanvasControlLayerAdapter | CanvasInpaintMaskAdapter | CanvasRegionalGuidanceAdapter | null
>(null);

export const RasterLayerAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const adapters = useSyncExternalStore(
    canvasManager.adapters.rasterLayers.subscribe,
    canvasManager.adapters.rasterLayers.getSnapshot
  );
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

RasterLayerAdapterGate.displayName = 'RasterLayerAdapterGate';

export const ControlLayerAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const adapters = useSyncExternalStore(
    canvasManager.adapters.controlLayers.subscribe,
    canvasManager.adapters.controlLayers.getSnapshot
  );
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

ControlLayerAdapterGate.displayName = 'ControlLayerAdapterGate';

export const InpaintMaskAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const adapters = useSyncExternalStore(
    canvasManager.adapters.inpaintMasks.subscribe,
    canvasManager.adapters.inpaintMasks.getSnapshot
  );
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

InpaintMaskAdapterGate.displayName = 'InpaintMaskAdapterGate';

export const RegionalGuidanceAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const adapters = useSyncExternalStore(
    canvasManager.adapters.regionMasks.subscribe,
    canvasManager.adapters.regionMasks.getSnapshot
  );
  const adapter = useMemo(() => {
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

RegionalGuidanceAdapterGate.displayName = 'RegionalGuidanceAdapterGate';

export const useEntityAdapter = ():
  | CanvasRasterLayerAdapter
  | CanvasControlLayerAdapter
  | CanvasInpaintMaskAdapter
  | CanvasRegionalGuidanceAdapter => {
  const adapter = useContext(EntityAdapterContext);
  assert(adapter, 'useEntityAdapter must be used within a CanvasRasterLayerAdapterGate');
  return adapter;
};
