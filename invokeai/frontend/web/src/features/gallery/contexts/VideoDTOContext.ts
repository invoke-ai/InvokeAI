import { createContext, useContext } from 'react';
import type { VideoDTO } from 'services/api/types';
import { assert } from 'tsafe';

const VideoDTOContext = createContext<VideoDTO | null>(null);

export const VideoDTOContextProvider = VideoDTOContext.Provider;

export const useVideoDTOContext = () => {
  const dto = useContext(VideoDTOContext);
  assert(dto !== null, 'useVideoDTOContext must be used within VideoDTOContextProvider');
  return dto;
};
