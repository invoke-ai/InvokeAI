import { createContext, useContext } from 'react';
import type { CanvasProjectDTO } from 'services/api/types';
import { assert } from 'tsafe';

const CanvasProjectDTOContext = createContext<CanvasProjectDTO | null>(null);

export const CanvasProjectDTOContextProvider = CanvasProjectDTOContext.Provider;

export const useCanvasProjectDTOContext = () => {
  const dto = useContext(CanvasProjectDTOContext);
  assert(dto !== null, 'useCanvasProjectDTOContext must be used within CanvasProjectDTOContextProvider');
  return dto;
};
