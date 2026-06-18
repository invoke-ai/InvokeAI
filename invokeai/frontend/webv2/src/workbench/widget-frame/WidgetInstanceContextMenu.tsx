import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { XIcon } from 'lucide-react';

import type { WidgetEnableMenuItem } from './WidgetEnableMenu';

export interface WidgetInstanceContextMenuTarget {
  item: WidgetEnableMenuItem;
  x: number;
  y: number;
}

interface WidgetInstanceContextMenuProps {
  target: WidgetInstanceContextMenuTarget | null;
  isRemoveDisabled?: (item: WidgetEnableMenuItem) => boolean;
  removeDisabledLabel?: string;
  onClose: () => void;
  onRemove: (item: WidgetEnableMenuItem) => void;
}

export const WidgetInstanceContextMenu = ({
  isRemoveDisabled,
  onClose,
  onRemove,
  removeDisabledLabel = 'Required',
  target,
}: WidgetInstanceContextMenuProps) => (
  <Menu.Root
    key={target?.item.id ?? 'closed'}
    lazyMount
    open={target !== null}
    positioning={{
      getAnchorRect: () => (target ? { height: 1, width: 1, x: target.x, y: target.y } : null),
      placement: 'bottom-start',
    }}
    unmountOnExit
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <Portal>
      <Menu.Positioner>
        {target ? (
          <Menu.Content minW="12rem">
            <Menu.Item
              value="remove-widget"
              disabled={isRemoveDisabled?.(target.item) === true}
              _disabled={{ opacity: 0.4 }}
              onClick={() => {
                if (isRemoveDisabled?.(target.item) !== true) {
                  onRemove(target.item);
                }
              }}
            >
              <Icon as={XIcon} boxSize="3.5" />
              <Menu.ItemText>Remove {target.item.label}</Menu.ItemText>
              {isRemoveDisabled?.(target.item) === true ? (
                <Text color="fg.subtle" fontSize="2xs" ms="auto">
                  {removeDisabledLabel}
                </Text>
              ) : null}
            </Menu.Item>
          </Menu.Content>
        ) : null}
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
