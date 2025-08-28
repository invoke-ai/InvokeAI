import { createContext, useContext } from 'react';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

const ImageDTOContext = createContext<ImageDTO | null>(null);

export const ImageDTOContextProvider = ImageDTOContext.Provider;

export const useImageDTOContext = () => {
  const imageDTO = useContext(ImageDTOContext);
  assert(imageDTO !== null, 'useImageDTOContext must be used within ImageDTOContextProvider');
  return imageDTO;
};
