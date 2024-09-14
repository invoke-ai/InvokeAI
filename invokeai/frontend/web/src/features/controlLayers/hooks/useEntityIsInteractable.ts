import { useStore } from '@nanostores/react';
import { $true } from 'app/store/nanostores/util';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';

export const useIsEntityInteractable = (entityIdentifier: CanvasEntityIdentifier) => {
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isInteractable = useStore(adapter?.$isInteractable ?? $true);

  return !isBusy && isInteractable;
};
