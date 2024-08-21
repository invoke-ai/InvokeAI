import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

const EntityMaskAdapterContext = createContext<CanvasMaskAdapter | null>(null);

export const EntityMaskAdapterProviderGate = memo(({ children }: PropsWithChildren) => {
  const entityIdentifier = useEntityIdentifierContext();
  const canvasManager = useCanvasManager();
  const adapter = useMemo(() => {
    if (entityIdentifier.type === 'inpaint_mask') {
      return canvasManager.inpaintMaskAdapters.get(entityIdentifier.id) ?? null;
    } else if (entityIdentifier.type === 'regional_guidance') {
      return canvasManager.regionalGuidanceAdapters.get(entityIdentifier.id) ?? null;
    }
    assert(false, 'EntityMaskAdapterProviderGate must be used with a valid EntityIdentifierContext');
  }, [canvasManager, entityIdentifier]);

  if (!canvasManager) {
    return null;
  }

  return <EntityMaskAdapterContext.Provider value={adapter}>{children}</EntityMaskAdapterContext.Provider>;
});

EntityMaskAdapterProviderGate.displayName = 'EntityMaskAdapterProviderGate';

export const useEntityMaskAdapter = (): CanvasMaskAdapter => {
  const adapter = useContext(EntityMaskAdapterContext);

  assert(adapter, 'useEntityMaskAdapter must be used within a EntityLayerAdapterProviderGate');

  return adapter;
};
