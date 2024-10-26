import { useAppSelector } from 'app/store/storeHooks';
import { buildSelectHasObjects } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIsEmpty = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectHasObjects = useMemo(() => buildSelectHasObjects(entityIdentifier), [entityIdentifier]);
  const hasObjects = useAppSelector(selectHasObjects);

  return !hasObjects;
};
