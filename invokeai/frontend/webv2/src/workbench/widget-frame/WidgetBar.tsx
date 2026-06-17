import type { RegisteredWidget, WidgetIconId, WidgetId, WidgetRegion } from '@workbench/types';

import { Flex, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Row, Tooltip } from '@workbench/components/ui';
import { WidgetIcon } from '@workbench/iconResolver';
import { CheckIcon, MoreHorizontalIcon } from 'lucide-react';

export interface WidgetBarItem {
  id: WidgetId;
  label: string;
  iconId: WidgetIconId;
  status?: RegisteredWidget['status'];
  failureMessage?: string;
  isEnabled: boolean;
}

interface WidgetBarProps {
  side: 'left' | 'right';
  region: Exclude<WidgetRegion, 'bottom' | 'center'>;
  activeId: WidgetId | null;
  railItems: WidgetBarItem[];
  menuItems: WidgetBarItem[];
  onSelect: (widgetId: WidgetId) => void;
  onToggle: (widgetId: WidgetId) => void;
}

/**
 * A widget bar (left / right rail).
 *
 * Phase 1 renders the rail and its slots; the widget registry, manifests, and
 * real views land in Phase 3. Each slot is a labelled button so the placeholders
 * are accessible and ready to bind to registered widgets later.
 */
export const WidgetBar = ({ activeId, menuItems, onSelect, onToggle, railItems, region, side }: WidgetBarProps) => (
  <Flex
    align="center"
    as="nav"
    bg="bg.subtle"
    borderColor="border.subtle"
    borderRightWidth={side === 'left' ? '1px' : '0'}
    borderLeftWidth={side === 'right' ? '1px' : '0'}
    direction="column"
    flexShrink={0}
    gap="1.5"
    pt="1.5"
    w="12"
  >
    {railItems.map((item) => (
      <WidgetSlot
        key={item.id}
        item={item}
        isActive={item.id === activeId}
        tooltipPlacement={side === 'left' ? 'right' : 'left'}
        onSelect={onSelect}
      />
    ))}
    <WidgetMenuButton region={region} items={menuItems} onToggle={onToggle} />
  </Flex>
);

const WidgetSlot = ({
  item,
  isActive,
  onSelect,
  tooltipPlacement,
}: {
  item: WidgetBarItem;
  isActive: boolean;
  onSelect: (widgetId: WidgetId) => void;
  tooltipPlacement: 'left' | 'right';
}) => {
  const tooltipLabel = item.failureMessage ? `${item.label}: ${item.failureMessage}` : item.label;
  const isDisabled = item.status === 'disabled';

  return (
    <Tooltip
      showArrow
      closeDelay={80}
      content={tooltipLabel}
      openDelay={250}
      positioning={{ placement: tooltipPlacement }}
    >
      <Row
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
      >
        <WidgetIcon iconId={item.iconId} boxSize="5" />
      </Row>
    </Tooltip>
  );
};

const WidgetMenuButton = ({
  items,
  onToggle,
  region,
}: {
  items: WidgetBarItem[];
  onToggle: (widgetId: WidgetId) => void;
  region: Exclude<WidgetRegion, 'bottom' | 'center'>;
}) => (
  <Menu.Root positioning={{ placement: region === 'left' ? 'right-start' : 'left-start' }}>
    <Menu.Trigger asChild>
      <Flex
        align="center"
        aria-label={`${region === 'left' ? 'Left' : 'Right'} widget visibility`}
        as="button"
        color="fg"
        h="9"
        justify="center"
        rounded="md"
        transition="background 0.12s ease, color 0.12s ease"
        w="9"
        _hover={{ bg: 'bg.muted' }}
      >
        <Icon as={MoreHorizontalIcon} boxSize="5" />
      </Flex>
    </Menu.Trigger>
    <Portal>
      <Menu.Positioner>
        <Menu.Content minW="12rem">
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
              Widgets
            </Menu.ItemGroupLabel>
            {items.map((item) => (
              <Menu.Item
                key={item.id}
                role="menuitemcheckbox"
                aria-checked={item.isEnabled}
                value={item.id}
                closeOnSelect={false}
                disabled={item.status === 'disabled'}
                _disabled={{ opacity: 0.4 }}
                onClick={() => onToggle(item.id)}
              >
                <Icon as={CheckIcon} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
                <Menu.ItemText>{item.label}</Menu.ItemText>
                {item.failureMessage ? (
                  <Text color="fg.subtle" fontSize="2xs" ms="auto">
                    Failed
                  </Text>
                ) : null}
              </Menu.Item>
            ))}
          </Menu.ItemGroup>
        </Menu.Content>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
