import { useCanvasManagerSafe } from 'features/controlLayers/hooks/useCanvasManager';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import type { CanvasEntityAdapterFromType } from 'features/controlLayers/konva/CanvasEntity/types';
import type { CanvasEntityIdentifier, CanvasEntityType } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useMemo, useSyncExternalStore } from 'react';
import { assert } from 'tsafe';

const EntityAdapterContext = createContext<
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance
  | null
>(null);

// Stable empty map instance to prevent infinite loops
const EMPTY_MAP = new Map();
const NOOP_SUBSCRIBE = () => () => {};
const GET_EMPTY_MAP = () => EMPTY_MAP;

export const RasterLayerAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManagerSafe();
  const entityIdentifier = useEntityIdentifierContext();
  
  const subscribe = useMemo(
    () => canvasManager?.adapters.rasterLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshot = useMemo(
    () => canvasManager?.adapters.rasterLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const adapters = useSyncExternalStore(subscribe, getSnapshot);
  
  const adapter = useMemo(() => {
    if (!canvasManager) {
      return null;
    }
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id, canvasManager]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

RasterLayerAdapterGate.displayName = 'RasterLayerAdapterGate';

export const ControlLayerAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManagerSafe();
  const entityIdentifier = useEntityIdentifierContext();
  
  const subscribe = useMemo(
    () => canvasManager?.adapters.controlLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshot = useMemo(
    () => canvasManager?.adapters.controlLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const adapters = useSyncExternalStore(subscribe, getSnapshot);
  
  const adapter = useMemo(() => {
    if (!canvasManager) {
      return null;
    }
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id, canvasManager]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

ControlLayerAdapterGate.displayName = 'ControlLayerAdapterGate';

export const InpaintMaskAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManagerSafe();
  const entityIdentifier = useEntityIdentifierContext();
  
  const subscribe = useMemo(
    () => canvasManager?.adapters.inpaintMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshot = useMemo(
    () => canvasManager?.adapters.inpaintMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const adapters = useSyncExternalStore(subscribe, getSnapshot);
  
  const adapter = useMemo(() => {
    if (!canvasManager) {
      return null;
    }
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id, canvasManager]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

InpaintMaskAdapterGate.displayName = 'InpaintMaskAdapterGate';

export const RegionalGuidanceAdapterGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useCanvasManagerSafe();
  const entityIdentifier = useEntityIdentifierContext();
  
  const subscribe = useMemo(
    () => canvasManager?.adapters.regionMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshot = useMemo(
    () => canvasManager?.adapters.regionMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const adapters = useSyncExternalStore(subscribe, getSnapshot);
  
  const adapter = useMemo(() => {
    if (!canvasManager) {
      return null;
    }
    return adapters.get(entityIdentifier.id) ?? null;
  }, [adapters, entityIdentifier.id, canvasManager]);

  if (!adapter) {
    return null;
  }

  return <EntityAdapterContext.Provider value={adapter}>{children}</EntityAdapterContext.Provider>;
});

export const useEntityAdapterContext = <T extends CanvasEntityType | undefined = CanvasEntityType>(
  type?: T
): CanvasEntityAdapterFromType<T extends undefined ? CanvasEntityType : T> => {
  const adapter = useContext(EntityAdapterContext);
  assert(adapter, 'useEntityIdentifier must be used within a EntityIdentifierProvider');
  if (type) {
    assert(adapter.entityIdentifier.type === type, 'useEntityIdentifier must be used with the correct type');
  }
  return adapter as CanvasEntityAdapterFromType<T extends undefined ? CanvasEntityType : T>;
};

RegionalGuidanceAdapterGate.displayName = 'RegionalGuidanceAdapterGate';

export const useEntityAdapterSafe = (
  entityIdentifier: CanvasEntityIdentifier | null
):
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance
  | null => {
  const canvasManager = useCanvasManagerSafe();
  
  const subscribeRegion = useMemo(
    () => canvasManager?.adapters.regionMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotRegion = useMemo(
    () => canvasManager?.adapters.regionMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeRaster = useMemo(
    () => canvasManager?.adapters.rasterLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotRaster = useMemo(
    () => canvasManager?.adapters.rasterLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeControl = useMemo(
    () => canvasManager?.adapters.controlLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotControl = useMemo(
    () => canvasManager?.adapters.controlLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeInpaint = useMemo(
    () => canvasManager?.adapters.inpaintMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotInpaint = useMemo(
    () => canvasManager?.adapters.inpaintMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const regionalGuidanceAdapters = useSyncExternalStore(subscribeRegion, getSnapshotRegion);
  const rasterLayerAdapters = useSyncExternalStore(subscribeRaster, getSnapshotRaster);
  const controlLayerAdapters = useSyncExternalStore(subscribeControl, getSnapshotControl);
  const inpaintMaskAdapters = useSyncExternalStore(subscribeInpaint, getSnapshotInpaint);

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

export const useAllEntityAdapters = () => {
  const canvasManager = useCanvasManagerSafe();
  
  const subscribeRegion = useMemo(
    () => canvasManager?.adapters.regionMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotRegion = useMemo(
    () => canvasManager?.adapters.regionMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeRaster = useMemo(
    () => canvasManager?.adapters.rasterLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotRaster = useMemo(
    () => canvasManager?.adapters.rasterLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeControl = useMemo(
    () => canvasManager?.adapters.controlLayers.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotControl = useMemo(
    () => canvasManager?.adapters.controlLayers.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const subscribeInpaint = useMemo(
    () => canvasManager?.adapters.inpaintMasks.subscribe ?? NOOP_SUBSCRIBE,
    [canvasManager]
  );
  
  const getSnapshotInpaint = useMemo(
    () => canvasManager?.adapters.inpaintMasks.getSnapshot ?? GET_EMPTY_MAP,
    [canvasManager]
  );
  
  const regionalGuidanceAdapters = useSyncExternalStore(subscribeRegion, getSnapshotRegion);
  const rasterLayerAdapters = useSyncExternalStore(subscribeRaster, getSnapshotRaster);
  const controlLayerAdapters = useSyncExternalStore(subscribeControl, getSnapshotControl);
  const inpaintMaskAdapters = useSyncExternalStore(subscribeInpaint, getSnapshotInpaint);
  
  const allEntityAdapters = useMemo(() => {
    return [
      ...Array.from(rasterLayerAdapters.values()),
      ...Array.from(controlLayerAdapters.values()),
      ...Array.from(inpaintMaskAdapters.values()),
      ...Array.from(regionalGuidanceAdapters.values()),
    ];
  }, [controlLayerAdapters, inpaintMaskAdapters, rasterLayerAdapters, regionalGuidanceAdapters]);

  return allEntityAdapters;
};