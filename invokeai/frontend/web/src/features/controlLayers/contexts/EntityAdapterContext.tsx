import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useMemo, useSyncExternalStore } from 'react';
import { assert } from 'tsafe';

const EntityAdapterContext = createContext<
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance
  | null
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

export const useEntityAdapterSafe = (
  entityIdentifier: CanvasEntityIdentifier | null
):
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance
  | null => {
  const canvasManager = useCanvasManager();
  const regionalGuidanceAdapters = useSyncExternalStore(
    canvasManager.adapters.regionMasks.subscribe,
    canvasManager.adapters.regionMasks.getSnapshot
  );
  const rasterLayerAdapters = useSyncExternalStore(
    canvasManager.adapters.rasterLayers.subscribe,
    canvasManager.adapters.rasterLayers.getSnapshot
  );
  const controlLayerAdapters = useSyncExternalStore(
    canvasManager.adapters.controlLayers.subscribe,
    canvasManager.adapters.controlLayers.getSnapshot
  );
  const inpaintMaskAdapters = useSyncExternalStore(
    canvasManager.adapters.inpaintMasks.subscribe,
    canvasManager.adapters.inpaintMasks.getSnapshot
  );

  const adapter = useMemo(() => {
    if (!entityIdentifier) {
      return null;
    }
    if (entityIdentifier.type === 'raster_layer') {
      return rasterLayerAdapters.get(entityIdentifier.id) ?? null;
    }
    if (entityIdentifier.type === 'control_layer') {
      return controlLayerAdapters.get(entityIdentifier.id) ?? null;
    }
    if (entityIdentifier.type === 'inpaint_mask') {
      return inpaintMaskAdapters.get(entityIdentifier.id) ?? null;
    }
    if (entityIdentifier.type === 'regional_guidance') {
      return regionalGuidanceAdapters.get(entityIdentifier.id) ?? null;
    }
    return null;
  }, [controlLayerAdapters, entityIdentifier, inpaintMaskAdapters, rasterLayerAdapters, regionalGuidanceAdapters]);

  return adapter;
};

export const useEntityAdapter = (
  entityIdentifier: CanvasEntityIdentifier
):
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance => {
  const adapter = useEntityAdapterSafe(entityIdentifier);
  assert(adapter, 'useEntityAdapter must be used within a EntityAdapterContext');
  return adapter;
};
