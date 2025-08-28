import { createContext, useContext } from 'react';
import { type ImageDTO, isImageDTO, type VideoDTO } from 'services/api/types';
import { assert } from 'tsafe';

const ItemDTOContext = createContext<ImageDTO | VideoDTO | null>(null);

export const ItemDTOContextProvider = ItemDTOContext.Provider;

export const useItemDTOContext = () => {
  const itemDTO = useContext(ItemDTOContext);
  assert(itemDTO !== null, 'useItemDTOContext must be used within ItemDTOContextProvider');
  return itemDTO;
};

export const useItemDTOContextImageOnly = (): ImageDTO => {
  const itemDTO = useContext(ItemDTOContext);
  assert(itemDTO !== null, 'useItemDTOContext must be used within ItemDTOContextProvider');
  assert(isImageDTO(itemDTO), 'ItemDTO is not an image');
  return itemDTO as ImageDTO;
};
