import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { WidgetPlacementInstanceMeta, WidgetRegionItem } from '@workbench/widgetRegionViewModel';

import { Box, Flex, type SystemStyleObject } from '@chakra-ui/react';
import { verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Row } from '@platform/ui/Row';
import { Tooltip } from '@platform/ui/Tooltip';
import { useHighlightedRegion } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { type MouseEvent, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useWidgetIntentPreloadProps } from './useWidgetIntentPreload';
import { useWidgetSortable } from './useWidgetSortable';
import { WidgetEnableMenu, type WidgetEnableMenuItem } from './WidgetEnableMenu';
import { WidgetInstanceContextMenu, type WidgetInstanceContextMenuTarget } from './WidgetInstanceContextMenu';
import { WidgetStrip } from './WidgetStrip';

export type WidgetBarItem = WidgetRegionItem<WidgetPlacementInstanceMeta>;

const WIDGET_SLOT_DISABLED_PROPS = { opacity: 0.4 };

/** One tab group the rail lists: the whole left region, or one dock of the right rail. */
export interface WidgetBarGroup {
  activeId: WidgetInstanceId | null;
  dropState: WidgetRegionDropState;
  railItems: WidgetBarItem[];
  region: Exclude<WidgetRegion, 'bottom' | 'center'>;
}

interface WidgetBarProps {
  /** The region along the rail's inner border, whose outline covers that border while it shows. */
  edgeRegion: WidgetRegion;
  side: 'left' | 'right';
  groups: WidgetBarGroup[];
  menuItems: WidgetBarItem[];
  onSelect: (region: WidgetBarGroup['region'], instanceId: WidgetInstanceId) => void;
  onToggle: (item: WidgetBarItem) => void;
}

export const WidgetBar = ({ edgeRegion, groups, menuItems, onSelect, onToggle, side }: WidgetBarProps) => {
  const { t } = useTranslation();
  const isEdgeOutlined = useHighlightedRegion() === edgeRegion;
  const region = side;
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
    <Flex
      aria-label={t('widgets.visibilityLabel', {
        region: side === 'left' ? t('widgets.rail.create') : t('widgets.rail.inspect'),
      })}
      as="nav"
      bg="bg.subtle"
      borderColor={isEdgeOutlined ? 'transparent' : 'border.subtle'}
      borderRightWidth={side === 'left' ? '1px' : '0'}
      borderLeftWidth={side === 'right' ? '1px' : '0'}
      direction="column"
      flexShrink={0}
      pt="1"
      w="11"
      onContextMenu={openEnableMenu}
    >
      {groups.map((group, index) => (
        <RailGroup
          key={group.region}
          group={group}
          separated={
            index > 0 && group.railItems.length > 0 && groups.slice(0, index).some((g) => g.railItems.length > 0)
          }
          side={side}
          onContextMenu={openInstanceMenu}
          onSelect={onSelect}
        />
      ))}

      <WidgetEnableMenu
        contextTarget={enableMenuTarget}
        groupLabel={region === 'left' ? t('widgets.rail.createWidgets') : t('widgets.rail.inspectWidgets')}
        items={menuItems}
        positioning={positioning}
        trigger={trigger}
        triggerLabel={region === 'left' ? t('widgets.rail.createWidgets') : t('widgets.rail.inspectWidgets')}
        onContextClose={handleContextClose}
        onToggle={handleMenuToggle}
      />

      <WidgetInstanceContextMenu
        target={instanceMenuTarget}
        onClose={handleInstanceClose}
        onRemove={handleMenuToggle}
      />
    </Flex>
  );
};

/** One dock's slots in the rail: its own sortable strip and drop target. */
const RailGroup = ({
  group,
  onContextMenu,
  onSelect,
  separated,
  side,
}: {
  group: WidgetBarGroup;
  onContextMenu: (item: WidgetBarItem, event: MouseEvent) => void;
  onSelect: (region: WidgetBarGroup['region'], instanceId: WidgetInstanceId) => void;
  separated: boolean;
  side: 'left' | 'right';
}) => {
  const sortableInstanceIds = useMemo(
    () => group.railItems.filter((item) => !item.isFloating).map((item) => item.id),
    [group.railItems]
  );
  const select = useCallback(
    (instanceId: WidgetInstanceId) => onSelect(group.region, instanceId),
    [group.region, onSelect]
  );

  return (
    <WidgetStrip
      align="center"
      borderColor="border.subtle"
      borderTopWidth={separated ? '1px' : '0'}
      data-rail-group={group.region}
      direction="column"
      dropState={group.dropState}
      display={group.railItems.length === 0 && !group.dropState.isActive ? 'none' : undefined}
      minH={group.railItems.length === 0 ? '10' : undefined}
      pt={separated ? '1' : undefined}
      region={group.region}
      sortableInstanceIds={sortableInstanceIds}
      strategy={verticalListSortingStrategy}
    >
      {group.railItems.map((item) => (
        <WidgetSlot
          key={item.id}
          item={item}
          isActive={item.id === group.activeId}
          region={group.region}
          tooltipPlacement={side === 'left' ? 'right' : 'left'}
          onContextMenu={onContextMenu}
          onSelect={select}
        />
      ))}
    </WidgetStrip>
  );
};

/**
 * Use brand color on active icons over neutral fill for cross-theme contrast. Attribute selectors preserve active
 * styling on hover.
 */
export const WIDGET_ITEM_SX: SystemStyleObject = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  rounded: 'md',
  h: 9,
  w: 9,
  color: 'fg.muted',
  '&[aria-pressed="false"]:hover': {
    bg: 'bg.emphasized',
    color: 'fg',
  },
  '&[aria-pressed="true"]': {
    bg: 'bg.emphasized',
    color: 'brand.fg',
  },
  // A floating slot reads as a placeholder for the window: outlined, not filled.
  '&[data-floating]': {
    outline: '1px dashed',
    outlineColor: 'border.emphasized',
    outlineOffset: '-1px',
    color: 'fg.subtle',
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
  const { t } = useTranslation();
  const tooltipLabel = item.isFloating
    ? t('widgets.floating.railSlot', { label: item.label })
    : item.failureMessage
      ? `${item.label}: ${item.failureMessage}`
      : item.label;
  const isDisabled = item.status === 'disabled';
  const positioning = useMemo(() => ({ placement: tooltipPlacement }) as const, [tooltipPlacement]);

  const handleClick = useCallback(() => {
    if (!isDisabled) {
      onSelect(item.id);
    }
  }, [isDisabled, item.id, onSelect]);
  const intentPreloadProps = useWidgetIntentPreloadProps(item.widget, isDisabled);

  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);

  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({
    disabled: isDisabled || item.isFloating === true,
    instanceId: item.id,
    region,
    typeId: item.typeId,
  });

  return (
    <Tooltip showArrow closeDelay={80} content={tooltipLabel} openDelay={250} positioning={positioning}>
      <Box ref={setNodeRef} pb="1" style={style}>
        <Row
          {...dragHandleProps}
          css={WIDGET_ITEM_SX}
          aria-label={tooltipLabel}
          aria-disabled={isDisabled}
          aria-pressed={isActive}
          as="button"
          data-disabled={isDisabled ? '' : undefined}
          data-floating={item.isFloating ? '' : undefined}
          tabIndex={isDisabled ? -1 : undefined}
          {...intentPreloadProps}
          onClick={handleClick}
          onContextMenu={handleContextMenu}
        >
          <WidgetIcon icon={item.icon} boxSize="5" />
        </Row>
      </Box>
    </Tooltip>
  );
};
