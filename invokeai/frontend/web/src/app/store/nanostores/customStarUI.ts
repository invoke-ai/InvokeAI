import { MenuItemProps } from '@chakra-ui/react';
import { atom } from 'nanostores';

export type CustomStarUi = {
  on: {
    icon: MenuItemProps['icon'];
    text: string;
  };
  off: {
    icon: MenuItemProps['icon'];
    text: string;
  };
};
export const $customStarUI = atom<CustomStarUi | undefined>(undefined);
