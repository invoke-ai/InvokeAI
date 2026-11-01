import type { ReactNode, Ref } from 'react';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { SlidersHorizontalIcon } from 'lucide-react';

import { useMenuTriggerIds } from './galleryMenuIds';

const FILTER_MENU_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * The Gallery's "Filter & sort" menu, shared by the boards and media toolbars so their trigger and grouped menu
 * cannot drift apart. It stays open on selection so several groups can be set in one visit.
 */
export const GalleryFilterMenu = ({
  children,
  isFiltered,
  label,
  triggerLabel = label,
  triggerRef,
}: {
  /** `Menu.ItemGroup`s, separated by `Menu.Separator`. */
  children: ReactNode;
  /** A non-default filter brightens the trigger so filtering is apparent with the menu closed. */
  isFiltered: boolean;
  /** The tooltip, and the accessible name unless `triggerLabel` states more. */
  label: string;
  triggerLabel?: string;
  triggerRef?: Ref<HTMLButtonElement>;
}) => {
  const menuTriggerIds = useMenuTriggerIds();

  return (
    <Menu.Root closeOnSelect={false} ids={menuTriggerIds} positioning={FILTER_MENU_POSITIONING}>
      <Tooltip content={label} ids={menuTriggerIds}>
        <Menu.Trigger asChild>
          <IconButton
            ref={triggerRef}
            aria-label={triggerLabel}
            color={isFiltered ? 'fg' : 'fg.muted'}
            flexShrink={0}
            variant="ghost"
          >
            <Icon as={SlidersHorizontalIcon} boxSize="3.5" />
          </IconButton>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="12rem">{children}</MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
