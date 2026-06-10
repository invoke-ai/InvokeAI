import { Flex, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { PiCheckBold, PiDotsThreeBold } from 'react-icons/pi';

import { WidgetIcon } from '../iconResolver';
import type { RegisteredWidget, WidgetIconId, WidgetId, WidgetRegion } from '../types';
import { Tooltip } from './ui/Tooltip';

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
    bg="bg.surface"
    borderColor="border.subtle"
    borderRightWidth={side === 'left' ? '1px' : '0'}
    borderLeftWidth={side === 'right' ? '1px' : '0'}
    direction="column"
    flexShrink={0}
    gap="1.5"
    pt="2.5"
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

  return (
    <Tooltip
      showArrow
      closeDelay={80}
      content={tooltipLabel}
      openDelay={250}
      positioning={{ placement: tooltipPlacement }}
    >
      <Flex
        align="center"
        aria-label={item.label}
        aria-disabled={item.status === 'disabled'}
        aria-pressed={isActive}
        as="button"
        bg={isActive ? 'accent.widget' : 'transparent'}
        color={isActive ? 'accent.widgetFg' : 'fg.default'}
        h="9"
        justify="center"
        rounded="md"
        opacity={item.status === 'disabled' ? 0.4 : 1}
        pointerEvents={item.status === 'disabled' ? 'none' : 'auto'}
        transition="background 0.12s ease, color 0.12s ease"
        w="9"
        _hover={{ bg: isActive ? 'accent.widget' : 'bg.surfaceRaised' }}
        onClick={() => onSelect(item.id)}
      >
        <WidgetIcon iconId={item.iconId} boxSize="5" />
      </Flex>
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
        color="fg.default"
        h="9"
        justify="center"
        rounded="md"
        transition="background 0.12s ease, color 0.12s ease"
        w="9"
        _hover={{ bg: 'bg.surfaceRaised' }}
      >
        <Icon as={PiDotsThreeBold} boxSize="5" />
      </Flex>
    </Menu.Trigger>
    <Portal>
      <Menu.Positioner>
        <Menu.Content
          bg="bg.surfaceRaised"
          borderWidth="1px"
          borderColor="border.emphasis"
          color="fg.default"
          minW="12rem"
          rounded="lg"
          shadow="lg"
        >
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
                <Icon as={PiCheckBold} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
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
