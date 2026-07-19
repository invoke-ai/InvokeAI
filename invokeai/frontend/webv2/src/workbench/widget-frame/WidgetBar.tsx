import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { WidgetPlacementInstanceMeta, WidgetRegionItem } from '@workbench/widgetRegionViewModel';

import { Box, type SystemStyleObject } from '@chakra-ui/react';
import { verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Row, Tooltip } from '@platform/ui';
import { WidgetIcon } from '@workbench/iconResolver';
import { type MouseEvent, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useWidgetSortable } from './useWidgetSortable';
import { WidgetEnableMenu, type WidgetEnableMenuItem } from './WidgetEnableMenu';
import { WidgetInstanceContextMenu, type WidgetInstanceContextMenuTarget } from './WidgetInstanceContextMenu';
import { WidgetStrip } from './WidgetStrip';

export type WidgetBarItem = WidgetRegionItem<WidgetPlacementInstanceMeta>;

const WIDGET_SLOT_DISABLED_PROPS = { opacity: 0.4 };

interface WidgetBarProps {
  side: 'left' | 'right';
  region: Exclude<WidgetRegion, 'bottom' | 'center'>;
  activeId: WidgetInstanceId | null;
  dropState: WidgetRegionDropState;
  railItems: WidgetBarItem[];
  menuItems: WidgetBarItem[];
  onSelect: (instanceId: WidgetInstanceId) => void;
  onToggle: (item: WidgetBarItem) => void;
}

export const WidgetBar = ({
  activeId,
  dropState,
  menuItems,
  onSelect,
  onToggle,
  railItems,
  region,
  side,
}: WidgetBarProps) => {
  const { t } = useTranslation();
  const [enableMenuTarget, setEnableMenuTarget] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);

  const openEnableMenu = useCallback((event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  }, []);

  const openInstanceMenu = useCallback((item: WidgetBarItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  }, []);

  const sortableInstanceIds = useMemo(() => railItems.map((item) => item.id), [railItems]);

  const positioning = useMemo(
    () =>
      ({
        placement: region === 'left' ? 'right-start' : 'left-start',
      }) as const,
    [region]
  );

  const trigger = useMemo(() => ({ kind: 'rail', region }) as const, [region]);
  const handleContextClose = useCallback(() => setEnableMenuTarget(null), []);
  const handleMenuToggle = useCallback((item: WidgetEnableMenuItem) => onToggle(item as WidgetBarItem), [onToggle]);
  const handleInstanceClose = useCallback(() => setInstanceMenuTarget(null), []);

  return (
    <WidgetStrip
      align="center"
      as="nav"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderRightWidth={side === 'left' ? '1px' : '0'}
      borderLeftWidth={side === 'right' ? '1px' : '0'}
      direction="column"
      dropState={dropState}
      flexShrink={0}
      pt="1.5"
      region={region}
      sortableInstanceIds={sortableInstanceIds}
      strategy={verticalListSortingStrategy}
      w="12"
      onContextMenu={openEnableMenu}
    >
      {railItems.map((item) => (
        <WidgetSlot
          key={item.id}
          item={item}
          isActive={item.id === activeId}
          region={region}
          tooltipPlacement={side === 'left' ? 'right' : 'left'}
          onContextMenu={openInstanceMenu}
          onSelect={onSelect}
        />
      ))}

      <Box mt="1.5">
        <WidgetEnableMenu
          contextTarget={enableMenuTarget}
          groupLabel={t('widgets.groupLabel')}
          items={menuItems}
          positioning={positioning}
          trigger={trigger}
          triggerLabel={t('widgets.visibilityLabel', { region: region === 'left' ? 'Left' : 'Right' })}
          onContextClose={handleContextClose}
          onToggle={handleMenuToggle}
        />
      </Box>

      <WidgetInstanceContextMenu
        target={instanceMenuTarget}
        onClose={handleInstanceClose}
        onRemove={handleMenuToggle}
      />
    </WidgetStrip>
  );
};

const WIDGET_ITEM_SX: SystemStyleObject = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  rounded: 'md',
  h: 9,
  w: 9,
  '&[aria-pressed="false"]:hover': {
    bg: 'bg.emphasized',
  },
  _disabled: WIDGET_SLOT_DISABLED_PROPS,
};

const WidgetSlot = ({
  item,
  isActive,
  onContextMenu,
  onSelect,
  region,
  tooltipPlacement,
}: {
  item: WidgetBarItem;
  isActive: boolean;
  onContextMenu: (item: WidgetBarItem, event: MouseEvent) => void;
  onSelect: (instanceId: WidgetInstanceId) => void;
  region: WidgetRegion;
  tooltipPlacement: 'left' | 'right';
}) => {
  const tooltipLabel = item.failureMessage ? `${item.label}: ${item.failureMessage}` : item.label;
  const isDisabled = item.status === 'disabled';
  const positioning = useMemo(() => ({ placement: tooltipPlacement }) as const, [tooltipPlacement]);

  const handleClick = useCallback(() => {
    if (!isDisabled) {
      onSelect(item.id);
    }
  }, [isDisabled, item.id, onSelect]);

  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);

  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({
    disabled: isDisabled,
    instanceId: item.id,
    region,
    typeId: item.typeId,
  });

  return (
    <Tooltip showArrow closeDelay={80} content={tooltipLabel} openDelay={250} positioning={positioning}>
      <Box ref={setNodeRef} pb="1.5" style={style}>
        <Row
          {...dragHandleProps}
          css={WIDGET_ITEM_SX}
          active={isActive ? 'accent' : 'none'}
          aria-label={item.label}
          aria-disabled={isDisabled}
          aria-pressed={isActive}
          as="button"
          data-disabled={isDisabled ? '' : undefined}
          tabIndex={isDisabled ? -1 : undefined}
          onClick={handleClick}
          onContextMenu={handleContextMenu}
        >
          <WidgetIcon icon={item.icon} boxSize="5" />
        </Row>
      </Box>
    </Tooltip>
  );
};
