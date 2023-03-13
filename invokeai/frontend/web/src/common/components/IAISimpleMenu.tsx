import {
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  MenuProps,
  MenuListProps,
  MenuItemProps,
  IconButton,
  Button,
  IconButtonProps,
  ButtonProps,
} from '@chakra-ui/react';
import { memo, MouseEventHandler, ReactNode } from 'react';
import { MdArrowDropDown, MdArrowDropUp } from 'react-icons/md';

interface IAIMenuItem {
  item: ReactNode | string;
  onClick: MouseEventHandler<HTMLButtonElement> | undefined;
}

interface IAIMenuProps {
  menuType?: 'icon' | 'regular';
  buttonText?: string;
  iconTooltip?: string;
  isLazy?: boolean;
  menuItems: IAIMenuItem[];
  menuProps?: MenuProps;
  menuButtonProps?: IconButtonProps | ButtonProps;
  menuListProps?: MenuListProps;
  menuItemProps?: MenuItemProps;
}

const IAISimpleMenu = (props: IAIMenuProps) => {
  const {
    menuType = 'icon',
    iconTooltip,
    buttonText,
    isLazy = true,
    menuItems,
    menuProps,
    menuButtonProps,
    menuListProps,
    menuItemProps,
  } = props;

  const renderMenuItems = () => {
    const menuItemsToRender: ReactNode[] = [];
    menuItems.forEach((menuItem, index) => {
      menuItemsToRender.push(
        <MenuItem
          key={index}
          onClick={menuItem.onClick}
          fontSize="sm"
          {...menuItemProps}
        >
          {menuItem.item}
        </MenuItem>
      );
    });
    return menuItemsToRender;
  };

  return (
    <Menu {...menuProps} isLazy={isLazy}>
      {({ isOpen }) => (
        <>
          <MenuButton
            as={menuType === 'icon' ? IconButton : Button}
            tooltip={iconTooltip}
            aria-label={iconTooltip}
            icon={isOpen ? <MdArrowDropUp /> : <MdArrowDropDown />}
            paddingX={0}
            paddingY={menuType === 'regular' ? 2 : 0}
            {...menuButtonProps}
          >
            {menuType === 'regular' && buttonText}
          </MenuButton>
          <MenuList zIndex={15} padding={0} {...menuListProps}>
            {renderMenuItems()}
          </MenuList>
        </>
      )}
    </Menu>
  );
};

export default memo(IAISimpleMenu);
