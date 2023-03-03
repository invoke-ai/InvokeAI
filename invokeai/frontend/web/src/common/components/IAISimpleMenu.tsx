import {
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  MenuProps,
  MenuButtonProps,
  MenuListProps,
  MenuItemProps,
} from '@chakra-ui/react';
import { MouseEventHandler, ReactNode } from 'react';
import { MdArrowDropDown, MdArrowDropUp } from 'react-icons/md';
import IAIButton from './IAIButton';
import IAIIconButton from './IAIIconButton';

interface IAIMenuItem {
  item: ReactNode | string;
  onClick: MouseEventHandler<HTMLButtonElement> | undefined;
}

interface IAIMenuProps {
  menuType?: 'icon' | 'regular';
  buttonText?: string;
  iconTooltip?: string;
  menuItems: IAIMenuItem[];
  menuProps?: MenuProps;
  menuButtonProps?: MenuButtonProps;
  menuListProps?: MenuListProps;
  menuItemProps?: MenuItemProps;
}

export default function IAISimpleMenu(props: IAIMenuProps) {
  const {
    menuType = 'icon',
    iconTooltip,
    buttonText,
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
          fontSize="0.9rem"
          color="var(--text-color-secondary)"
          backgroundColor="var(--background-color-secondary)"
          _focus={{
            color: 'var(--text-color)',
            backgroundColor: 'var(--border-color)',
          }}
          {...menuItemProps}
        >
          {menuItem.item}
        </MenuItem>
      );
    });
    return menuItemsToRender;
  };

  return (
    <Menu {...menuProps}>
      {({ isOpen }) => (
        <>
          <MenuButton
            as={menuType === 'icon' ? IAIIconButton : IAIButton}
            tooltip={iconTooltip}
            icon={isOpen ? <MdArrowDropUp /> : <MdArrowDropDown />}
            padding={menuType === 'regular' ? '0 0.5rem' : 0}
            backgroundColor="var(--btn-base-color)"
            _hover={{
              backgroundColor: 'var(--btn-base-color-hover)',
            }}
            minWidth="1rem"
            minHeight="1rem"
            fontSize="1.5rem"
            {...menuButtonProps}
          >
            {menuType === 'regular' && buttonText}
          </MenuButton>
          <MenuList
            zIndex={15}
            padding={0}
            borderRadius="0.5rem"
            backgroundColor="var(--background-color-secondary)"
            color="var(--text-color-secondary)"
            borderColor="var(--border-color)"
            {...menuListProps}
          >
            {renderMenuItems()}
          </MenuList>
        </>
      )}
    </Menu>
  );
}
