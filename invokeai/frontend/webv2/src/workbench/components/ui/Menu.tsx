import { Menu } from '@chakra-ui/react';
import type { ComponentProps } from 'react';

type MenuContentProps = ComponentProps<typeof Menu.Content>;

/**
 * Menu.Content with the workbench popover surface styling applied, so every
 * menu (filter menus, context menus, action menus) renders the same surface.
 */
export const MenuContent = (props: MenuContentProps) => (
  <Menu.Content
    bg="bg.surfaceRaised"
    borderColor="border.emphasis"
    borderWidth="1px"
    color="fg.default"
    rounded="lg"
    shadow="lg"
    {...props}
  />
);
