import { Box, Flex, HStack, Icon, Menu, Portal, Tabs, Text } from '@chakra-ui/react';
import { PiCheckBold, PiDotsThreeBold } from 'react-icons/pi';

import { WidgetIcon } from '../iconResolver';
import { useWorkbench } from '../WorkbenchContext';
import type { CenterViewId, RegisteredWidget } from '../types';
import { getWidgetsForRegion } from '../widgetRegistry';
import { WidgetRenderer } from './WidgetRenderer';

interface CenterWidgetItem {
  id: Exclude<CenterViewId, 'preview'>;
  isEnabled: boolean;
  label: string;
  status: RegisteredWidget['status'];
  widget: RegisteredWidget;
}

const getCenterWidgetItems = (enabledWidgetIds: string[]): CenterWidgetItem[] =>
  getWidgetsForRegion('center').map((widget) => ({
    id: widget.manifest.id as Exclude<CenterViewId, 'preview'>,
    isEnabled: enabledWidgetIds.includes(widget.manifest.id),
    label: widget.manifest.labelText,
    status: widget.status,
    widget,
  }));

/** Center work area: the view tab strip plus the active registered center view. */
export const CenterArea = () => {
  const { activeProject, dispatch } = useWorkbench();
  const centerRegion = activeProject.widgetRegions.center;
  const centerWidgetItems = getCenterWidgetItems(centerRegion.enabledWidgetIds);
  const enabledCenterWidgetItems = centerWidgetItems.filter((item) => item.isEnabled && item.status === 'enabled');

  return (
    <Flex as="section" bg="bg.center" direction="column" flex="1" minH="0" minW="0">
      <HStack bg="bg.surfaceRaised" borderBottomWidth="1px" borderColor="border.subtle" h="10" px="1.5">
        <Tabs.Root
          value={centerRegion.activeWidgetId}
          variant="plain"
          onValueChange={(event) =>
            dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: event.value as CenterWidgetItem['id'] })
          }
        >
          <Tabs.List gap="1">
            {enabledCenterWidgetItems.map((item) => (
              <Tabs.Trigger
                key={item.id}
                value={item.id}
                color="fg.muted"
                disabled={item.status === 'disabled'}
                fontSize="xs"
                fontWeight="600"
                gap="1.5"
                h="7"
                px="3"
                rounded="md"
                _selected={{ bg: 'accent.active', color: 'accent.activeFg' }}
                _disabled={{ opacity: 0.4 }}
                _hover={{ color: 'fg.default', _selected: { color: 'accent.activeFg' } }}
              >
                <WidgetIcon iconId={item.widget.manifest.icon} boxSize="3.5" />
                {item.label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </Tabs.Root>
        <Box flex="1" />
        <CenterWidgetMenu
          enabledCount={centerRegion.enabledWidgetIds.length}
          items={centerWidgetItems}
          onToggle={(centerWidgetId) =>
            dispatch({ region: 'center', type: 'toggleRegionWidget', widgetId: centerWidgetId })
          }
        />
      </HStack>

      <Box flex="1" minH="0" position="relative">
        <CenterViewSlot activeWidgetId={centerRegion.activeWidgetId} items={enabledCenterWidgetItems} />
      </Box>
    </Flex>
  );
};

const CenterViewSlot = ({ activeWidgetId, items }: { activeWidgetId: string; items: CenterWidgetItem[] }) => {
  const widget = items.find((item) => item.id === activeWidgetId)?.widget;
  const View = widget?.manifest.view;

  if (!widget || widget.status !== 'enabled' || !View) {
    return <FallbackCenterView label="Center widget unavailable" />;
  }

  return <WidgetRenderer widget={widget} region="center" />;
};

const CenterWidgetMenu = ({
  enabledCount,
  items,
  onToggle,
}: {
  enabledCount: number;
  items: CenterWidgetItem[];
  onToggle: (centerWidgetId: CenterWidgetItem['id']) => void;
}) => (
  <Menu.Root positioning={{ placement: 'bottom-end' }}>
    <Menu.Trigger asChild>
      <Flex
        align="center"
        aria-label="Center widget menu"
        as="button"
        color="fg.default"
        h="7"
        justify="center"
        rounded="md"
        transition="background 0.12s ease, color 0.12s ease"
        w="7"
        _hover={{ bg: 'bg.surface' }}
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
              Center Widgets
            </Menu.ItemGroupLabel>
            {items.map((item) => (
              <Menu.Item
                key={item.id}
                role="menuitemcheckbox"
                aria-checked={item.isEnabled}
                value={item.id}
                closeOnSelect={false}
                disabled={item.status === 'disabled' || (item.isEnabled && enabledCount === 1)}
                _disabled={{ opacity: 0.4 }}
                onClick={() => onToggle(item.id)}
              >
                <Icon as={PiCheckBold} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
                <WidgetIcon iconId={item.widget.manifest.icon} boxSize="3.5" />
                <Menu.ItemText>{item.label}</Menu.ItemText>
                {item.isEnabled && enabledCount === 1 ? (
                  <Text color="fg.subtle" fontSize="2xs" ms="auto">
                    Required
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

const FallbackCenterView = ({ label }: { label: string }) => (
  <Flex align="center" h="full" justify="center" w="full">
    <Text color="fg.subtle" fontSize="sm" textTransform="capitalize">
      {label} view
    </Text>
  </Flex>
);
