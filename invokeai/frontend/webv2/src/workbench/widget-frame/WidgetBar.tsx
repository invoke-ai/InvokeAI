import type { WidgetInstanceId, WidgetRegion } from '@workbench/types';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { WidgetPlacementInstanceMeta, WidgetRegionItem } from '@workbench/widgetRegionViewModel';

import { Box } from '@chakra-ui/react';
import { verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Row, Tooltip } from '@workbench/components/ui';
import { WidgetIcon } from '@workbench/iconResolver';
import { type MouseEvent, useState } from 'react';

import { useWidgetSortable } from './useWidgetSortable';
import { WidgetEnableMenu } from './WidgetEnableMenu';
import { WidgetInstanceContextMenu, type WidgetInstanceContextMenuTarget } from './WidgetInstanceContextMenu';
import { WidgetStrip } from './WidgetStrip';

export type WidgetBarItem = WidgetRegionItem<WidgetPlacementInstanceMeta>;

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

/**
 * A widget bar (left / right rail).
 *
 * Phase 1 renders the rail and its slots; the widget registry, manifests, and
 * real views land in Phase 3. Each slot is a labelled button so the placeholders
 * are accessible and ready to bind to registered widgets later.
 */
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
  const [enableMenuTarget, setEnableMenuTarget] = useState<{ x: number; y: number } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);

  const openEnableMenu = (event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  };

  const openInstanceMenu = (item: WidgetBarItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  };

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
      sortableInstanceIds={railItems.map((item) => item.id)}
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
          groupLabel="Widgets"
          items={menuItems}
          positioning={{ placement: region === 'left' ? 'right-start' : 'left-start' }}
          trigger={{ kind: 'rail', region }}
          triggerLabel={`${region === 'left' ? 'Left' : 'Right'} widget visibility`}
          onContextClose={() => setEnableMenuTarget(null)}
          onToggle={(item) => onToggle(item as WidgetBarItem)}
        />
      </Box>
      <WidgetInstanceContextMenu
        target={instanceMenuTarget}
        onClose={() => setInstanceMenuTarget(null)}
        onRemove={(item) => onToggle(item as WidgetBarItem)}
      />
    </WidgetStrip>
  );
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
  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({
    disabled: isDisabled,
    instanceId: item.id,
    region,
    typeId: item.typeId,
  });

  return (
    <Tooltip
      showArrow
      closeDelay={80}
      content={tooltipLabel}
      openDelay={250}
      positioning={{ placement: tooltipPlacement }}
    >
      <Box ref={setNodeRef} pb="1.5" style={style}>
        <Row
          {...dragHandleProps}
          active={isActive ? 'accent' : 'none'}
          aria-label={item.label}
          aria-disabled={isDisabled}
          aria-pressed={isActive}
          as="button"
          data-disabled={isDisabled ? '' : undefined}
          h="9"
          justifyContent="center"
          rounded="md"
          tabIndex={isDisabled ? -1 : undefined}
          w="9"
          _disabled={{ opacity: 0.4 }}
          onClick={() => {
            if (!isDisabled) {
              onSelect(item.id);
            }
          }}
          onContextMenu={(event) => onContextMenu(item, event)}
        >
          <WidgetIcon icon={item.icon} boxSize="5" />
        </Row>
      </Box>
    </Tooltip>
  );
};
