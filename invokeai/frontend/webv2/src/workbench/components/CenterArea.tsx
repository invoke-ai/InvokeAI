import { Box, Flex, HStack, Icon, Menu, Portal, Tabs, Text } from '@chakra-ui/react';
import { PiCheckBold, PiDotsThreeBold } from 'react-icons/pi';

import { WidgetIcon } from '../iconResolver';
import { useWorkbench } from '../WorkbenchContext';
import type { RegisteredWidget, WidgetId } from '../types';
import { getWidgetsForRegion } from '../widgetRegistry';
import { useFocusRegionProps } from '../focusRegions';
import { WidgetRenderer } from './WidgetRenderer';

interface CenterWidgetItem {
  id: WidgetId;
  isEnabled: boolean;
  label: string;
  status: RegisteredWidget['status'];
  widget: RegisteredWidget;
}

const getCenterWidgetItems = (enabledWidgetIds: string[]): CenterWidgetItem[] =>
  getWidgetsForRegion('center').map((widget) => ({
    id: widget.manifest.id,
    isEnabled: enabledWidgetIds.includes(widget.manifest.id),
    label: widget.manifest.labelText,
    status: widget.status,
    widget,
  }));

/** Center work area: the view tab strip plus the active registered center view. */
export const CenterArea = () => {
  const { activeProject, dispatch } = useWorkbench();
  const focusRegionProps = useFocusRegionProps('center');
  const centerRegion = activeProject.widgetRegions.center;
  const centerWidgetItems = getCenterWidgetItems(centerRegion.enabledWidgetIds);
  const enabledCenterWidgetItems = centerWidgetItems.filter((item) => item.isEnabled && item.status === 'enabled');
  const centerViewItems = enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement !== 'toolbar');
  const centerToolbarItems = enabledCenterWidgetItems.filter(
    (item) => item.widget.manifest.centerPlacement === 'toolbar'
  );
  const activeCenterViewId = centerViewItems.some((item) => item.id === centerRegion.activeWidgetId)
    ? centerRegion.activeWidgetId
    : centerViewItems[0]?.id;

  return (
    <Flex as="section" bg="bg.center" direction="column" flex="1" minH="0" minW="0" {...focusRegionProps}>
      <HStack bg="bg.surfaceRaised" borderBottomWidth="1px" borderColor="border.subtle" h="10" px="1.5">
        <Tabs.Root
          value={activeCenterViewId}
          variant="line"
          h="full"
          w="full"
          onValueChange={(event) =>
            dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: event.value as CenterWidgetItem['id'] })
          }
        >
          <Tabs.List gap="1">
            {centerViewItems.map((item) => (
              <Tabs.Trigger key={item.id} value={item.id} disabled={item.status === 'disabled'} fontSize="xs" px="3">
                <WidgetIcon iconId={item.widget.manifest.icon} boxSize="3.5" />
                {item.label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </Tabs.Root>
        <Box flex="1" />
        {centerToolbarItems.map((item) => (
          <Flex key={item.id} align="center" h="full">
            <WidgetRenderer widget={item.widget} presentation="compact" region="center" />
          </Flex>
        ))}
        <CenterWidgetMenu
          enabledViewCount={centerViewItems.length}
          items={centerWidgetItems}
          onToggle={(centerWidgetId) =>
            dispatch({ region: 'center', type: 'toggleRegionWidget', widgetId: centerWidgetId })
          }
        />
      </HStack>

      <Box flex="1" minH="0" position="relative">
        <CenterViewSlot activeWidgetId={activeCenterViewId} items={centerViewItems} />
      </Box>
    </Flex>
  );
};

const CenterViewSlot = ({
  activeWidgetId,
  items,
}: {
  activeWidgetId: string | undefined;
  items: CenterWidgetItem[];
}) => {
  const widget = items.find((item) => item.id === activeWidgetId)?.widget;
  const View = widget?.manifest.view;

  if (!widget || widget.status !== 'enabled' || !View) {
    return <FallbackCenterView label="Center widget unavailable" />;
  }

  return <WidgetRenderer widget={widget} region="center" />;
};

const CenterWidgetMenu = ({
  enabledViewCount,
  items,
  onToggle,
}: {
  enabledViewCount: number;
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
            {items.map((item) => {
              const isView = item.widget.manifest.centerPlacement !== 'toolbar';
              const isRequiredView = isView && item.isEnabled && enabledViewCount === 1;

              return (
                <Menu.Item
                  key={item.id}
                  role="menuitemcheckbox"
                  aria-checked={item.isEnabled}
                  value={item.id}
                  closeOnSelect={false}
                  disabled={item.status === 'disabled' || isRequiredView}
                  _disabled={{ opacity: 0.4 }}
                  onClick={() => onToggle(item.id)}
                >
                  <Icon as={PiCheckBold} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
                  <WidgetIcon iconId={item.widget.manifest.icon} boxSize="3.5" />
                  <Menu.ItemText>{item.label}</Menu.ItemText>
                  {isRequiredView ? (
                    <Text color="fg.subtle" fontSize="2xs" ms="auto">
                      Required
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

const FallbackCenterView = ({ label }: { label: string }) => (
  <Flex align="center" h="full" justify="center" w="full">
    <Text color="fg.subtle" fontSize="sm" textTransform="capitalize">
      {label} view
    </Text>
  </Flex>
);
