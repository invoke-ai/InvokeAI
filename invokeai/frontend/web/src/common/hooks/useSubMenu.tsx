import type { MenuButtonProps, MenuItemProps, MenuListProps, MenuProps } from '@invoke-ai/ui-library';
import { Box, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useDisclosure } from 'common/hooks/useBoolean';
import type { FocusEventHandler, PointerEvent } from 'react';
import { useCallback } from 'react';
import { PiCaretRightBold } from 'react-icons/pi';
import { useDebouncedCallback } from 'use-debounce';

const offset: [number, number] = [0, 8];

type UseSubMenuReturn = {
  parentMenuItemProps: Partial<MenuItemProps>;
  menuProps: Partial<MenuProps>;
  menuButtonProps: Partial<MenuButtonProps>;
  menuListProps: Partial<MenuListProps>;
};

/**
 * A hook that provides the necessary props to create a sub-menu within a menu.
 *
 * The sub-menu should be wrapped inside a parent `MenuItem` component.
 *
 * Use SubMenuButtonContent to render a button with a label and a right caret icon.
 *
 * TODO(psyche): Add keyboard handling for sub-menu.
 *
 * @example
 * ```tsx
 * const SubMenuExample = () => {
 *   const subMenu = useSubMenu();
 *   return (
 *     <Menu>
 *       <MenuButton>Open Parent Menu</MenuButton>
 *       <MenuList>
 *         <MenuItem>Parent Item 1</MenuItem>
 *         <MenuItem>Parent Item 2</MenuItem>
 *         <MenuItem>Parent Item 3</MenuItem>
 *         <MenuItem {...subMenu.parentMenuItemProps} icon={<PiImageBold />}>
 *           <Menu {...subMenu.menuProps}>
 *             <MenuButton {...subMenu.menuButtonProps}>
 *               <SubMenuButtonContent label="Open Sub Menu" />
 *             </MenuButton>
 *             <MenuList {...subMenu.menuListProps}>
 *               <MenuItem>Sub Item 1</MenuItem>
 *               <MenuItem>Sub Item 2</MenuItem>
 *               <MenuItem>Sub Item 3</MenuItem>
 *             </MenuList>
 *           </Menu>
 *         </MenuItem>
 *       </MenuList>
 *     </Menu>
 *   );
 * };
 * ```
 */
export const useSubMenu = (): UseSubMenuReturn => {
  const subMenu = useDisclosure(false);
  const closeDebounced = useDebouncedCallback(subMenu.close, 300);
  const openAndCancelPendingClose = useCallback(() => {
    closeDebounced.cancel();
    subMenu.open();
  }, [closeDebounced, subMenu]);
  const toggleAndCancelPendingClose = useCallback(() => {
    if (subMenu.isOpen) {
      subMenu.close();
      return;
    } else {
      closeDebounced.cancel();
      subMenu.toggle();
    }
  }, [closeDebounced, subMenu]);
  const onBlurMenuList = useCallback<FocusEventHandler<HTMLDivElement>>(
    (e) => {
      // Don't trigger blur if focus is moving to a child element - e.g. from a sub-menu item to another sub-menu item
      if (e.currentTarget.contains(e.relatedTarget)) {
        closeDebounced.cancel();
        return;
      }
      subMenu.close();
    },
    [closeDebounced, subMenu]
  );
  const onParentMenuItemPointerLeave = useCallback(
    (e: PointerEvent<HTMLButtonElement>) => {
      if (e.pointerType === 'pen' || e.pointerType === 'touch') {
        return;
      }
      closeDebounced();
    },
    [closeDebounced]
  );

  return {
    parentMenuItemProps: {
      onClick: toggleAndCancelPendingClose,
      onPointerEnter: openAndCancelPendingClose,
      onPointerLeave: onParentMenuItemPointerLeave,
      closeOnSelect: false,
    },
    menuProps: {
      isOpen: subMenu.isOpen,
      onClose: subMenu.close,
      placement: 'right',
      offset: offset,
      closeOnBlur: false,
    },
    menuButtonProps: {
      as: Box,
      width: 'full',
      height: 'full',
    },
    menuListProps: {
      onPointerEnter: openAndCancelPendingClose,
      onPointerLeave: closeDebounced,
      onBlur: onBlurMenuList,
    },
  };
};

export const SubMenuButtonContent = ({ label }: { label: string }) => {
  return (
    <Flex w="full" h="full" flexDir="row" justifyContent="space-between" alignItems="center">
      <Text>{label}</Text>
      <Icon as={PiCaretRightBold} />
    </Flex>
  );
};
