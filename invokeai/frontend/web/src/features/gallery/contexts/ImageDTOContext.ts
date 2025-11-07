import { createContext, useContext } from 'react';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

const ImageDTOCOntext = createContext<ImageDTO | null>(null);

export const ImageDTOContextProvider = ImageDTOCOntext.Provider;

export const useImageDTOContext = () => {
  const dto = useContext(ImageDTOCOntext);
  assert(dto !== null, 'useItemDTOContext must be used within ItemDTOContextProvider');
  return dto;
};
