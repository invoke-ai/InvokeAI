import type {
  MenuButtonProps as ChakraMenuButtonProps,
  MenuDividerProps as ChakraMenuDividerProps,
  MenuGroupProps as ChakraMenuGroupProps,
  MenuItemOptionProps as ChakraMenuItemOptionProps,
  MenuItemProps as ChakraMenuItemProps,
  MenuListProps as ChakraMenuListProps,
  MenuOptionGroupProps as ChakraMenuOptionGroupProps,
  MenuProps as ChakraMenuProps,
} from '@chakra-ui/react';

export type InvMenuProps = ChakraMenuProps;
export type InvMenuButtonProps = ChakraMenuButtonProps;
export type InvMenuListProps = ChakraMenuListProps;
export type InvMenuItemProps = ChakraMenuItemProps & {
  isDestructive?: boolean;
  isLoading?: boolean;
};
export type InvMenuItemOptionProps = ChakraMenuItemOptionProps;
export type InvMenuGroupProps = ChakraMenuGroupProps;
export type InvMenuOptionGroupProps = ChakraMenuOptionGroupProps;
export type InvMenuDividerProps = ChakraMenuDividerProps;
