import { Box, Flex, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { PiCheckBold, PiDotsThreeBold } from 'react-icons/pi';

import { WidgetIcon } from '../iconResolver';
import type { RegisteredWidget, WidgetId } from '../types';
import { getWidgetsForRegion } from '../widgetRegistry';
import { useWorkbench } from '../WorkbenchContext';
import { WidgetRenderer } from './WidgetRenderer';
import { Tooltip } from './ui/Tooltip';

interface BottomWidgetItem {
  failureMessage?: string;
  id: WidgetId;
  isEnabled: boolean;
  isExpandable: boolean;
  label: string;
  status: RegisteredWidget['status'];
  widget: RegisteredWidget;
}

export const StatusBar = () => {
  const { activeProject, dispatch } = useWorkbench();
  const bottomRegion = activeProject.widgetRegions.bottom;
  const items: BottomWidgetItem[] = getWidgetsForRegion('bottom').map((widget) => ({
    failureMessage: widget.failure?.message,
    id: widget.manifest.id,
    isEnabled: bottomRegion.enabledWidgetIds.includes(widget.manifest.id),
    isExpandable: widget.manifest.bottomPanel !== 'tooltip',
    label: widget.manifest.labelText,
    status: widget.status,
    widget,
  }));
  const compactItems = items.filter((item) => item.isEnabled && item.status === 'enabled');

  return (
    <Flex
      align="center"
      as="footer"
      bg="bg.surface"
      borderTopWidth="1px"
      borderColor="border.subtle"
      color="fg.muted"
      flexShrink={0}
      gap="1.5"
      h="6"
      px="2"
      w="full"
    >
      {compactItems.map((item) => (
        <CompactBottomWidget
          key={item.id}
          item={item}
          isActive={item.isExpandable && item.id === bottomRegion.activeWidgetId && !bottomRegion.isCollapsed}
          onSelect={(widgetId) => dispatch({ region: 'bottom', type: 'selectRegionWidget', widgetId })}
        />
      ))}
      <Box flex="1" />
      <BottomWidgetMenu
        items={items}
        onToggle={(widgetId) => dispatch({ region: 'bottom', type: 'toggleRegionWidget', widgetId })}
      />
    </Flex>
  );
};

const CompactBottomWidget = ({
  isActive,
  item,
  onSelect,
}: {
  isActive: boolean;
  item: BottomWidgetItem;
  onSelect: (widgetId: WidgetId) => void;
}) => {
  const View = item.widget.manifest.view;

  if (!View) {
    return null;
  }

  const content = (
    <Box
      aria-label={item.label}
      aria-pressed={isActive}
      bg={isActive ? 'accent.active' : 'transparent'}
      color={isActive ? 'accent.activeFg' : 'fg.muted'}
      cursor={item.isExpandable ? 'pointer' : 'default'}
      display="flex"
      h="full"
      alignItems="center"
      role={item.isExpandable ? 'button' : undefined}
      rounded="sm"
      tabIndex={item.isExpandable ? 0 : undefined}
      _hover={{
        bg: isActive ? 'accent.active' : 'bg.surfaceRaised',
        color: isActive ? 'accent.activeFg' : 'fg.default',
      }}
      onClick={() => {
        if (item.isExpandable) {
          onSelect(item.id);
        }
      }}
      onKeyDown={(event) => {
        if (item.isExpandable && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          onSelect(item.id);
        }
      }}
    >
      <WidgetRenderer widget={item.widget} presentation="compact" region="bottom" />
    </Box>
  );

  if (item.isExpandable) {
    return (
      <Tooltip
        closeDelay={80}
        content={item.failureMessage ? `${item.label}: ${item.failureMessage}` : item.label}
        openDelay={250}
        positioning={{ placement: 'top-start' }}
        showArrow
      >
        {content}
      </Tooltip>
    );
  }

  return (
    <Tooltip
      closeDelay={80}
      content={<WidgetRenderer widget={item.widget} presentation="tooltip" region="bottom" />}
      openDelay={250}
      positioning={{ placement: 'top-start' }}
      showArrow
    >
      {content}
    </Tooltip>
  );
};

const BottomWidgetMenu = ({
  items,
  onToggle,
}: {
  items: BottomWidgetItem[];
  onToggle: (widgetId: WidgetId) => void;
}) => (
  <Menu.Root positioning={{ placement: 'top-end' }}>
    <Menu.Trigger asChild>
      <Flex
        align="center"
        aria-label="Bottom widget visibility"
        as="button"
        color="fg.default"
        h="5"
        justify="center"
        rounded="sm"
        transition="background 0.12s ease, color 0.12s ease"
        w="5"
        _hover={{ bg: 'bg.surfaceRaised' }}
      >
        <Icon as={PiDotsThreeBold} boxSize="4" />
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
              Bottom Widgets
            </Menu.ItemGroupLabel>
            {items.map((item) => {
              return (
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
                  <WidgetIcon iconId={item.widget.manifest.icon} boxSize="3.5" />
                  <Menu.ItemText>{item.label}</Menu.ItemText>
                  {item.failureMessage ? (
                    <Text color="fg.subtle" fontSize="2xs" ms="auto">
                      Failed
                    </Text>
                  ) : null}
                </Menu.Item>
              );
            })}
          </Menu.ItemGroup>
        </Menu.Content>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
