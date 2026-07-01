import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

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

const REMOVE_DISABLED_PROPS = { opacity: 0.4 };

export const WidgetInstanceContextMenu = ({
  isRemoveDisabled,
  onClose,
  onRemove,
  removeDisabledLabel = 'Required',
  target,
}: WidgetInstanceContextMenuProps) => {
  const { t } = useTranslation();
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => (target ? { height: 1, width: 1, x: target.x, y: target.y } : null),
      placement: 'bottom-start' as const,
    }),
    [target]
  );
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );
  const isDisabled = target ? isRemoveDisabled?.(target.item) === true : false;
  const handleRemove = useCallback(() => {
    if (target && isRemoveDisabled?.(target.item) !== true) {
      onRemove(target.item);
    }
  }, [isRemoveDisabled, onRemove, target]);

  return (
    <Menu.Root
      key={target?.item.id ?? 'closed'}
      lazyMount
      open={target !== null}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Menu.Positioner>
          {target ? (
            <Menu.Content minW="12rem">
              <Menu.Item
                value="remove-widget"
                disabled={isDisabled}
                _disabled={REMOVE_DISABLED_PROPS}
                onClick={handleRemove}
              >
                <Icon as={XIcon} boxSize="3.5" />
                <Menu.ItemText>{t('widgets.removeWidget', { label: target.item.label })}</Menu.ItemText>
                {isDisabled ? (
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
};
