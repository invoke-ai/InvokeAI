import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useIsEntitySelected = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectedEntityIdentifier = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier);
  const isSelected = useMemo(() => {
    return selectedEntityIdentifier?.id === entityIdentifier.id;
  }, [selectedEntityIdentifier, entityIdentifier.id]);

  return isSelected;
};
