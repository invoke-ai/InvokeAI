import { Box, Flex, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { CheckIcon, MoreHorizontalIcon } from 'lucide-react';

import { WidgetIcon } from '../iconResolver';
import { useActiveProjectSelector, useWorkbenchDispatch } from '../WorkbenchContext';
import type { RegisteredWidget, WidgetId } from '../types';
import { getWidgetsForRegion } from '../widgetRegistry';
import { useFocusRegionProps } from '../focusRegions';
import { WidgetRenderer } from './WidgetRenderer';
import { IconButton } from './ui/Button';
import { Tabs } from './ui/Tabs';

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
  const centerRegion = useActiveProjectSelector((project) => project.widgetRegions.center);
  const dispatch = useWorkbenchDispatch();
  const focusRegionProps = useFocusRegionProps('center');
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
    <Flex as="section" bg="bg.inset" direction="column" flex="1" minH="0" minW="0" {...focusRegionProps}>
      <HStack bg="bg.muted" borderBottomWidth="1px" borderColor="border.subtle" h="10" px="1.5">
        <Tabs.Root
          value={activeCenterViewId}
          h="full"
          w="full"
          onValueChange={(event) =>
            dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: event.value as CenterWidgetItem['id'] })
          }
        >
          <Tabs.List>
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
}) => {
  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <IconButton aria-label="Center widget menu" size="xs" variant="ghost">
          <Icon as={MoreHorizontalIcon} boxSize="4" />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content minW="12rem">
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
                    <Icon as={CheckIcon} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
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
};

const FallbackCenterView = ({ label }: { label: string }) => (
  <Flex align="center" h="full" justify="center" w="full">
    <Text color="fg.subtle" fontSize="sm" textTransform="capitalize">
      {label} view
    </Text>
  </Flex>
);
