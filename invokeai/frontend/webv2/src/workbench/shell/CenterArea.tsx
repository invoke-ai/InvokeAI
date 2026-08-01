import type {
  PlacedWidgetRegionItem,
  WidgetPlacementInstanceMeta,
  WidgetRegionItem,
} from '@workbench/widgetRegionViewModel';

import { Box, Flex, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { useModelLoads } from '@features/models';
import { getProjectQueueIndicatorState, type QueueProgressBarState } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { IconButton, MenuContent } from '@platform/ui';
import { QueueTabBackgroundProgress } from '@workbench/components/QueueProgressIndicator';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import {
  WidgetChromeSlotById,
  WidgetRendererById,
  WidgetSourceLockBadge,
  useWidgetIntentPreloadProps,
  type WidgetEnableMenuItem,
} from '@workbench/widget-frame';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import {
  createWidgetRegionViewModelFromState,
  getWidgetRegionItems,
  isRequiredCenterView,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CheckIcon, ChevronDownIcon, XIcon } from 'lucide-react';
import { Suspense, useCallback, useMemo, useRef, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

type CenterWidgetItem = PlacedWidgetRegionItem<WidgetPlacementInstanceMeta>;

const CENTER_MENU_POSITIONING = { placement: 'bottom-start' } as const;
const CENTER_PREFERRED_REGIONS = ['center'] as const;

/**
 * The chrome floats over the work surface, so widgets whose own content is
 * top-anchored (the gallery toolbar) clear it with this inset instead of the
 * region reserving a row. Full-bleed surfaces — canvas, workflow, preview —
 * ignore it and run their content edge to edge beneath the islands.
 */
const CENTER_CHROME_INSET_STYLE = { '--wb-center-chrome-inset': '3rem' } as React.CSSProperties;

/**
 * Center work area: a full-bleed registered center view with two floating
 * chrome islands over it — the view selector (plus the active widget's own
 * label slot) at the start, the active widget's actions at the end.
 */
export const CenterArea = () => {
  const { t } = useTranslation();
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const centerRegion = useActiveProjectSelector((project) => project.widgetRegions.center);
  const invocation = useActiveProjectSelector((project) => project.invocation);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const { widgets } = useWorkbenchCommands();
  const getWidgetLabel = useCallback(
    (manifest: Parameters<typeof resolveWidgetLabel>[0]) => resolveWidgetLabel(manifest, t),
    [t]
  );

  const centerRegionViewModel = createWidgetRegionViewModelFromState({
    getWidgetLabel,
    region: 'center',
    regionState: centerRegion,
    widgetInstances: placementProject.widgetInstances,
    widgets: getWidgetsForRegion('center'),
  });

  const centerWidgetMenuItems = useMemo(() => getWidgetRegionItems(centerRegionViewModel), [centerRegionViewModel]);

  const enabledCenterWidgetItems = useMemo(
    () => centerRegionViewModel.placedItems.filter((item) => item.status === 'enabled'),
    [centerRegionViewModel.placedItems]
  );

  const centerViewItems = useMemo(
    () => enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement !== 'toolbar'),
    [enabledCenterWidgetItems]
  );

  const centerToolbarItems = useMemo(
    () => enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement === 'toolbar'),
    [enabledCenterWidgetItems]
  );

  const activeCenterViewId = centerViewItems.some((item) => item.id === centerRegion.activeInstanceId)
    ? centerRegion.activeInstanceId
    : centerViewItems[0]?.id;
  const activeItem = centerViewItems.find((item) => item.id === activeCenterViewId);
  const focusRegionProps = useFocusRegionProps('center');

  const openCenterWidget = useCallback(
    (item: WidgetEnableMenuItem) =>
      openWidgetPlacement({
        widgets,
        getWidgetsForRegion,
        options: {
          createNew: item.allowMultiple,
          preferredRegions: CENTER_PREFERRED_REGIONS,
        },
        typeId: item.typeId,
      }),
    [widgets]
  );

  const selectCenterView = useCallback(
    (instanceId: string) =>
      revealWidgetPlacement({
        instanceId: instanceId as CenterWidgetItem['id'],
        project: placementProject,
        region: 'center',
        widgets,
      }),
    [placementProject, widgets]
  );

  const closeActiveCenterView = useCallback(() => {
    if (activeCenterViewId) {
      closeWidgetPlacement({
        widgets,
        getWidgetById,
        instanceId: activeCenterViewId,
        project: placementProject,
        region: 'center',
      });
    }
  }, [activeCenterViewId, placementProject, widgets]);

  const isActiveViewRequired = activeItem
    ? isRequiredCenterView(activeItem as WidgetRegionItem, centerViewItems.length)
    : true;

  // Widgets that can be added: every center-capable type that has no instance
  // placed here yet (`availableItems` already accounts for `allowMultiple`).
  const availableCenterItems = useMemo(
    () => centerWidgetMenuItems.filter((item) => !item.isEnabled && item.status !== 'disabled'),
    [centerWidgetMenuItems]
  );

  const baseIndicatorState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount: modelLoads.length,
    progress: null,
    queueItems,
  });

  const progress = useQueueItemProgress(baseIndicatorState.runningQueueItemId ?? '');

  const indicatorState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount: modelLoads.length,
    progress,
    queueItems,
  });
  const showProgress = indicatorState.hasOpenQueueWork && activeItem?.typeId === invocation.sourceId;

  return (
    <Flex
      as="section"
      bg="bg"
      direction="column"
      flex="1"
      minH="0"
      minW="0"
      style={CENTER_CHROME_INSET_STYLE}
      {...focusRegionProps}
    >
      <Box flex="1" minH="0" position="relative">
        <Box aria-label={t('widgets.centerRegion')} h="full" role="region">
          {activeItem ? <CenterViewSlot item={activeItem} /> : <FallbackCenterView label="Center widget unavailable" />}
        </Box>

        <Flex
          data-hotkey-widget-instance-id={activeItem?.id}
          data-hotkey-widget-region="center"
          data-hotkey-widget-type-id={activeItem?.typeId}
          gap="2"
          insetX="2"
          justify="space-between"
          pointerEvents="none"
          position="absolute"
          top="2"
          zIndex="1"
        >
          <ChromeIsland>
            <CenterViewMenu
              activeItem={activeItem}
              availableItems={availableCenterItems}
              isCloseDisabled={isActiveViewRequired}
              items={centerViewItems}
              progressState={indicatorState.progressState}
              showProgress={showProgress}
              onClose={closeActiveCenterView}
              onOpen={openCenterWidget}
              onSelect={selectCenterView}
            />
            {activeItem ? (
              <>
                <WidgetSourceLockBadge typeId={activeItem.typeId} />
                <Suspense fallback={null}>
                  <WidgetChromeSlotById instanceId={activeItem.id} slot="label" widget={activeItem.widget} />
                </Suspense>
              </>
            ) : null}
          </ChromeIsland>

          {activeItem && activeItem.widget.manifest.chrome?.header !== 'hidden' ? (
            <ChromeIsland>
              {centerToolbarItems.map((toolbarItem) => (
                <WidgetRendererById
                  key={toolbarItem.id}
                  instanceId={toolbarItem.id}
                  widget={toolbarItem.widget}
                  presentation="compact"
                  region="center"
                />
              ))}
              <Suspense fallback={null}>
                <WidgetChromeSlotById instanceId={activeItem.id} slot="actions" widget={activeItem.widget} />
              </Suspense>
            </ChromeIsland>
          ) : null}
        </Flex>
      </Box>
    </Flex>
  );
};

/**
 * One floating chrome group. Islands carry the same `bg.subtle` surface as the
 * top bar, the widget rails, and the side panels, so the chrome reads as one
 * material wherever it appears. Pointer events are re-enabled per island so the
 * gap between them stays click-through.
 */
const ChromeIsland = ({ children }: { children: ReactNode }) => (
  <HStack
    bg="bg.subtle"
    borderColor="border.subtle"
    borderWidth="1px"
    gap="0.5"
    h="8"
    minW="0"
    pointerEvents="auto"
    position="relative"
    px="1"
    rounded="md"
    shadow="sm"
  >
    {children}
  </HStack>
);

const CenterViewMenu = ({
  activeItem,
  availableItems,
  isCloseDisabled,
  items,
  onClose,
  onOpen,
  onSelect,
  progressState,
  showProgress,
}: {
  activeItem?: CenterWidgetItem;
  availableItems: WidgetEnableMenuItem[];
  isCloseDisabled: boolean;
  items: CenterWidgetItem[];
  onClose: () => void;
  onOpen: (item: WidgetEnableMenuItem) => void;
  onSelect: (instanceId: string) => void;
  progressState: QueueProgressBarState;
  showProgress: boolean;
}) => {
  const { t } = useTranslation();
  const label = activeItem?.label ?? t('widgets.centerViewEmpty');
  const triggerRef = useRef<HTMLButtonElement>(null);
  const selectAndRestoreFocus = useCallback(
    (instanceId: string) => {
      onSelect(instanceId);
      requestAnimationFrame(() => {
        triggerRef.current?.focus({ preventScroll: true });
      });
    },
    [onSelect]
  );

  return (
    <Menu.Root positioning={CENTER_MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton
          ref={triggerRef}
          aria-label={t('widgets.centerViewLabel', { label })}
          color="fg"
          minW="0"
          overflow="hidden"
          position="relative"
          px="2"
          size="2xs"
          variant="ghost"
        >
          {showProgress ? <QueueTabBackgroundProgress state={progressState} zIndex="-1" /> : null}
          <HStack gap="2" minW="0" position="relative" zIndex="1">
            {activeItem ? <WidgetIcon icon={activeItem.widget.manifest.icon} boxSize="3.5" /> : null}
            <Text fontSize="xs" fontWeight="700" truncate>
              {label}
            </Text>
            <ChevronDownIcon size={12} />
          </HStack>
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                {t('widgets.centerViews')}
              </Menu.ItemGroupLabel>
              {items.map((item) => (
                <CenterViewMenuRow
                  key={item.id}
                  isActive={item.id === activeItem?.id}
                  item={item}
                  onSelect={selectAndRestoreFocus}
                />
              ))}
            </Menu.ItemGroup>

            {availableItems.length > 0 ? (
              <>
                <Menu.Separator borderColor="border.subtle" />
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                    {t('widgets.centerViewsAdd')}
                  </Menu.ItemGroupLabel>
                  {availableItems.map((item) => (
                    <CenterViewMenuRow key={item.id} item={item} onOpen={onOpen} />
                  ))}
                </Menu.ItemGroup>
              </>
            ) : null}

            {activeItem ? (
              <>
                <Menu.Separator borderColor="border.subtle" />
                <Menu.Item disabled={isCloseDisabled} value="close-center-view" onClick={onClose}>
                  <Icon as={XIcon} boxSize="3.5" color="fg.subtle" />
                  <Menu.ItemText fontSize="xs">
                    {t('widgets.centerViewClose', { label: activeItem.label })}
                  </Menu.ItemText>
                </Menu.Item>
              </>
            ) : null}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/**
 * One row of the selector. Open views are radio items so the active one is
 * announced as checked; rows that would open a new view stay plain menu items,
 * since they are an action rather than a choice within the current set.
 */
const CenterViewMenuRow = ({
  isActive,
  item,
  onOpen,
  onSelect,
}: {
  isActive?: boolean;
  item: CenterWidgetItem | WidgetEnableMenuItem;
  onOpen?: (item: WidgetEnableMenuItem) => void;
  onSelect?: (instanceId: string) => void;
}) => {
  const handleClick = useCallback(
    () => (onSelect ? onSelect(item.id) : onOpen?.(item as WidgetEnableMenuItem)),
    [item, onOpen, onSelect]
  );
  const intentPreloadProps = useWidgetIntentPreloadProps(item.widget, item.status === 'disabled');

  return (
    <Menu.Item
      role={onSelect ? 'menuitemradio' : undefined}
      aria-checked={onSelect ? isActive : undefined}
      value={item.id}
      disabled={item.status === 'disabled'}
      {...intentPreloadProps}
      onClick={handleClick}
    >
      <Icon as={CheckIcon} boxSize="3" opacity={isActive ? 1 : 0} />
      <WidgetIcon icon={item.icon} boxSize="3.5" />
      <Menu.ItemText fontSize="xs">{item.label}</Menu.ItemText>
    </Menu.Item>
  );
};

const CenterViewSlot = ({ item }: { item: CenterWidgetItem }) => {
  if (item.widget.status !== 'enabled') {
    return <FallbackCenterView label="Center widget unavailable" />;
  }

  return <WidgetRendererById instanceId={item.instance.id} widget={item.widget} region="center" />;
};

const FallbackCenterView = ({ label }: { label: string }) => (
  <Flex align="center" h="full" justify="center" w="full">
    <Text color="fg.subtle" fontSize="sm" textTransform="capitalize">
      {label} view
    </Text>
  </Flex>
);
