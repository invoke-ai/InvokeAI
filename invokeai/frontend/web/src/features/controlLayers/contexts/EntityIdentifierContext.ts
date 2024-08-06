import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { createContext, useContext } from 'react';
import { assert } from 'tsafe';

export const EntityIdentifierContext = createContext<CanvasEntityIdentifier | null>(null);

export const useEntityIdentifierContext = (): CanvasEntityIdentifier => {
  const entityIdentifier = useContext(EntityIdentifierContext);
  assert(entityIdentifier, 'useEntityIdentifier must be used within a EntityIdentifierProvider');
  return entityIdentifier;
};
