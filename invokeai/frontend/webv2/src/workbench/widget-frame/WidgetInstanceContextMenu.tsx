import type { WidgetRegion } from '@workbench/layoutContracts';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui/Menu';
import { DOCK_DESTINATION_ICONS, resolveDockLabel } from '@workbench/widgetLabels';
import { ArrowLeftToLineIcon, ArrowRightToLineIcon, XIcon } from 'lucide-react';
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
  /** Set, with `onDock`, when the target is a floating window's marker: the region docking returns it to. */
  dockRegion?: WidgetRegion;
  /** With `onSetAlignment`, offers moving the widget between the strip's two clusters. */
  isAlignedEnd?: (item: WidgetEnableMenuItem) => boolean;
  isRemoveDisabled?: (item: WidgetEnableMenuItem) => boolean;
  removeDisabledLabel?: string;
  onClose: () => void;
  onDock?: (item: WidgetEnableMenuItem) => void;
  onRemove: (item: WidgetEnableMenuItem) => void;
  onSetAlignment?: (item: WidgetEnableMenuItem, align: 'start' | 'end') => void;
}

const REMOVE_DISABLED_PROPS = { opacity: 0.4 };

export const WidgetInstanceContextMenu = ({
  dockRegion,
  isAlignedEnd,
  isRemoveDisabled,
  onClose,
  onDock,
  onRemove,
  onSetAlignment,
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
  const isEnd = target ? isAlignedEnd?.(target.item) === true : false;
  const handleToggleAlignment = useCallback(() => {
    if (target) {
      onSetAlignment?.(target.item, isEnd ? 'start' : 'end');
    }
  }, [isEnd, onSetAlignment, target]);
  const handleDock = useCallback(() => {
    if (target) {
      onDock?.(target.item);
    }
  }, [onDock, target]);
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
            <MenuContent minW="12rem">
              {onDock && dockRegion ? (
                <Menu.Item value="dock-widget" onClick={handleDock}>
                  <Icon as={DOCK_DESTINATION_ICONS[dockRegion]} boxSize="3.5" />
                  <Menu.ItemText>{resolveDockLabel(dockRegion, t)}</Menu.ItemText>
                </Menu.Item>
              ) : null}
              {onSetAlignment && target.item.isEnabled ? (
                <Menu.Item value="toggle-alignment" onClick={handleToggleAlignment}>
                  <Icon as={isEnd ? ArrowLeftToLineIcon : ArrowRightToLineIcon} boxSize="3.5" />
                  <Menu.ItemText>{isEnd ? t('widgets.moveToLeftSide') : t('widgets.moveToRightSide')}</Menu.ItemText>
                </Menu.Item>
              ) : null}
              <Menu.Item
                value="remove-widget"
                disabled={isDisabled}
                _disabled={REMOVE_DISABLED_PROPS}
                onClick={handleRemove}
              >
                <Icon as={XIcon} boxSize="3.5" />
                <Menu.ItemText>{t('widgets.removeWidget', { label: target.item.label })}</Menu.ItemText>
                {isDisabled ? (
                  <Text color="fg.subtle" fontSize="xs" ms="auto">
                    {removeDisabledLabel}
                  </Text>
                ) : null}
              </Menu.Item>
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
