import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectEntityIdentifierBelowThisOne, selectSelectedCanvas } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIdentifierBelowThisOne = <T extends CanvasEntityIdentifier>(entityIdentifier: T): T | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectSelectedCanvas, (canvas) => {
        const nextEntity = selectEntityIdentifierBelowThisOne(canvas, entityIdentifier);
        if (!nextEntity) {
          return null;
        }
        return getEntityIdentifier(nextEntity);
      }),
    [entityIdentifier]
  );
  const entityIdentifierBelowThisOne = useAppSelector(selector);

  return entityIdentifierBelowThisOne as T | null;
};
