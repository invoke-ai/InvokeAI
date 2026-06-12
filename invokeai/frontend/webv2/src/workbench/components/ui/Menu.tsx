import { Menu } from '@chakra-ui/react';
import type { ComponentProps } from 'react';

type MenuContentProps = ComponentProps<typeof Menu.Content>;

/**
 * Menu.Content passthrough. The workbench popover chrome (surface, stroke,
 * radius, shadow) is applied globally by the `menu` slot-recipe override in
 * `theme/recipes.ts`; this wrapper only exists as the single import point
 * for future menu-wide behavior.
 */
export const MenuContent = (props: MenuContentProps) => <Menu.Content {...props} />;
