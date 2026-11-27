import type { CanvasEntityIdentifier, CanvasEntityType } from 'features/controlLayers/store/types';
import { createContext, useContext } from 'react';
import { assert } from 'tsafe';

export const EntityIdentifierContext = createContext<CanvasEntityIdentifier | null>(null);

export const useEntityIdentifierContext = <T extends CanvasEntityType | undefined = CanvasEntityType>(
  type?: T
): CanvasEntityIdentifier<T extends undefined ? CanvasEntityType : T> => {
  const entityIdentifier = useContext(EntityIdentifierContext);
  assert(entityIdentifier, 'useEntityIdentifier must be used within a EntityIdentifierProvider');
  if (type) {
    assert(entityIdentifier.type === type, 'useEntityIdentifier must be used with the correct type');
  }
  return entityIdentifier as CanvasEntityIdentifier<T extends undefined ? CanvasEntityType : T>;
};
