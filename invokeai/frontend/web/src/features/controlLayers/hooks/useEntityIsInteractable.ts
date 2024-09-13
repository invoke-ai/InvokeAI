import { useStore } from '@nanostores/react';
import { useEntityAdapter } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';

export const useIsEntityInteractable = (entityIdentifier: CanvasEntityIdentifier) => {
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapter(entityIdentifier);
  const isInteractable = useStore(adapter.$isInteractable);

  return !isBusy && isInteractable;
};
