import type { RegisteredWidget } from '@workbench/widgetContracts';
import type { PlacedWidgetRegionItem, WidgetPlacementInstanceMeta } from '@workbench/widgetRegionViewModel';

import { Box, Flex, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Button, IconButton, MenuContent } from '@platform/ui';
import { ClippedTooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useFocusRegionProps, useWorkbenchFocus } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import {
  WidgetChromeSlotById,
  WidgetIdentityIcon,
  WidgetRendererById,
  WidgetSourceLockBadge,
  useWidgetIntentPreloadProps,
  type WidgetEnableMenuItem,
} from '@workbench/widget-frame';
import { areWidgetRenderInstancesEqual } from '@workbench/widget-frame/widgetRenderInstance';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { addWidgetPlacement, closeWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { createWidgetRegionViewModelFromState, getWidgetRegionItems } from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import {
  useActiveLayoutSelector,
  useActiveProjectId,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { takesLastCenterView } from '@workbench/workspace/structureReducer';
import { useCanEditStructure } from '@workbench/workspace/useStructureLock';
import { getActiveAwayStage, goBackToStage } from '@workbench/workspace/workspaceNavigation';
import { ArrowLeftIcon, CheckIcon, ChevronDownIcon, XIcon } from 'lucide-react';
import { Activity, Suspense, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { ChromeIsland, getChromeIslandMinWidth } from './ChromeIsland';
import {
  areInstanceIdListsEqual,
  getActiveInstanceIdsOutside,
  useMountedInstanceIds,
  withoutInstancesShownElsewhere,
} from './useMountedInstanceIds';

type CenterWidgetItem = PlacedWidgetRegionItem<WidgetPlacementInstanceMeta>;

const CENTER_MENU_POSITIONING = { placement: 'bottom-start' } as const;
const CENTER_REGIONS = ['center'] as const;

const CENTER_CHROME_INSET_STYLE = { '--wb-center-chrome-inset': '3rem' } as React.CSSProperties;
// Reserve a usable center width; side panels shrink to honor this floor.
const CENTER_MIN_WIDTH = '20rem';
/** The view menu keeps its icon, a few letters of its name, and its chevron; the row scrolls past that. */
const VIEW_MENU_MIN_WIDTH = '5rem';
/** Below this center width the view menu shows only its icon and chevron. Matches the Preview's compact density. */
const CENTER_COMPACT_MAX_WIDTH_PX = 560;
/** The chrome row's inset on each side (`insetX="2"`); the query container is the row, not the whole center. */
const CHROME_INSET_PX = 8;
const CHROME_CONTAINER_CSS = { containerName: 'center-chrome', containerType: 'inline-size' } as const;
const COMPACT_CHROME = `@container center-chrome (width < ${CENTER_COMPACT_MAX_WIDTH_PX - 2 * CHROME_INSET_PX}px)`;
/** The view menu also minimises when the start island's room is below this, sparing the widget's label. */
const START_ISLAND_ROOM_CSS = {
  containerName: 'center-start',
  containerType: 'inline-size',
  display: 'flex',
  flex: '1',
  minW: '0',
} as const;
const COMPACT_START = '@container center-start (width < 18rem)';
const compactViewMenu = <T extends object>(styles: T) => ({ [COMPACT_CHROME]: styles, [COMPACT_START]: styles });
const VIEW_MENU_TRIGGER_CSS = compactViewMenu({ flexShrink: 0, minW: '8', px: '0' });
const VIEW_MENU_CONTENT_CSS = { gap: '2', ...compactViewMenu({ gap: '1' }) };
const VIEW_MENU_LABEL_CSS = compactViewMenu({ display: 'none' });
/** Back to the stage compacts with the view menu, to its icon, which stays on screen at the island's floor. */
const BACK_COMPACT_WIDTH = '2rem';
const BACK_TRIGGER_CSS = compactViewMenu({ flexShrink: 0, minW: BACK_COMPACT_WIDTH, px: '0' });
/** The start island gives way first, down to this width: the view menu at its floor (and Back, while Away). */
const getEndIslandMaxWidth = (startControlsWidth: string) =>
  `calc(100% - ${getChromeIslandMinWidth(startControlsWidth)} - {spacing.2})`;
const END_ISLAND_MAX_WIDTH = getEndIslandMaxWidth(VIEW_MENU_MIN_WIDTH);
const END_ISLAND_MAX_WIDTH_AWAY = getEndIslandMaxWidth(`(${VIEW_MENU_MIN_WIDTH} + ${BACK_COMPACT_WIDTH} + 2px)`);

export const CenterArea = () => {
  const { t } = useTranslation();
  const placementProject = useWorkbenchSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const centerRegion = useActiveLayoutSelector((layout) => layout.widgetRegions.center);
  const commands = useWorkbenchCommands();
  const { widgets } = commands;
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const awayStageId = useWorkbenchSelector(getActiveAwayStage, Object.is);
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
  const awayStage = awayStageId === null ? undefined : centerViewItems.find((item) => item.id === awayStageId);
  const goBack = useCallback(
    () => goBackToStage({ commands, focus, getWidgetsForRegion, queries }),
    [commands, focus, queries]
  );
  const projectId = useActiveProjectId();
  const activeIdsElsewhere = useActiveLayoutSelector(
    (layout) => getActiveInstanceIdsOutside(layout.widgetRegions, 'center', layout.floatingWidgets),
    areInstanceIdListsEqual
  );
  const mountedCenterIds = withoutInstancesShownElsewhere(
    useMountedInstanceIds(activeCenterViewId, projectId),
    activeCenterViewId,
    activeIdsElsewhere
  );
  const focusRegionProps = useFocusRegionProps('center');

  const openCenterWidget = useCallback(
    (item: WidgetEnableMenuItem) =>
      addWidgetPlacement({
        createNew: item.allowMultiple,
        getWidgetsForRegion,
        regions: CENTER_REGIONS,
        typeId: item.typeId,
        widgets,
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
        availability: queries.getAvailability(),
        instanceId: activeCenterViewId,
        project: placementProject,
        region: 'center',
        widgets,
      });
    }
  }, [activeCenterViewId, placementProject, queries, widgets]);

  const isActiveViewRequired = activeItem
    ? takesLastCenterView(placementProject, activeItem.id, queries.getAvailability())
    : true;

  const canEditStructure = useCanEditStructure();
  const availableCenterItems = useMemo(
    () =>
      canEditStructure ? centerWidgetMenuItems.filter((item) => !item.isEnabled && item.status !== 'disabled') : [],
    [canEditStructure, centerWidgetMenuItems]
  );

  return (
    <Flex
      as="section"
      role="group"
      aria-label={activeItem?.label ?? t('widgets.centerViewEmpty')}
      bg="bg"
      direction="column"
      flex="1"
      minH="0"
      minW={CENTER_MIN_WIDTH}
      style={CENTER_CHROME_INSET_STYLE}
      {...focusRegionProps}
    >
      <Box flex="1" minH="0" position="relative">
        <Box aria-label={t('widgets.centerRegion')} h="full" role="region">
          {activeCenterViewId !== undefined && mountedCenterIds.length > 0 ? (
            mountedCenterIds.map((instanceId) => (
              <Activity key={instanceId} mode={instanceId === activeCenterViewId ? 'visible' : 'hidden'}>
                <KeptCenterViewSlot instanceId={instanceId} />
              </Activity>
            ))
          ) : (
            <FallbackCenterView />
          )}
        </Box>

        <Flex
          css={CHROME_CONTAINER_CSS}
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
          <Box css={START_ISLAND_ROOM_CSS}>
            <ChromeIsland>
              {awayStage ? <BackToStageButton label={awayStage.label} onBack={goBack} /> : null}
              <CenterViewMenu
                activeItem={activeItem}
                availableItems={availableCenterItems}
                isCloseDisabled={isActiveViewRequired}
                items={centerViewItems}
                onClose={canEditStructure ? closeActiveCenterView : undefined}
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
          </Box>

          {activeItem && activeItem.widget.manifest.chrome?.header !== 'hidden' ? (
            <ChromeIsland flexShrink={0} maxW={awayStage ? END_ISLAND_MAX_WIDTH_AWAY : END_ISLAND_MAX_WIDTH}>
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

/** The header's first control while Away; Escape in the center does the same. */
const BackToStageButton = ({ label, onBack }: { label: string; onBack: () => void }) => {
  const { t } = useTranslation();
  const text = t('shell.workspace.backToStage', { stage: label });
  const labelRef = useRef<HTMLParagraphElement>(null);
  // Named by the tooltip while its label is truncated or, compact, hidden.
  const isLabelClipped = useCallback(() => {
    const label = labelRef.current;

    return !label || label.getClientRects().length === 0 || label.scrollWidth > label.clientWidth;
  }, []);

  return (
    <ClippedTooltip content={text} isClipped={isLabelClipped}>
      <Button
        aria-label={text}
        color="fg"
        // The button recipe refuses to shrink; the label truncates to the floor, then the control compacts.
        css={BACK_TRIGGER_CSS}
        flexShrink={1}
        gap="1.5"
        minW={VIEW_MENU_MIN_WIDTH}
        overflow="hidden"
        px="2"
        size="sm"
        variant="ghost"
        onClick={onBack}
      >
        <Icon as={ArrowLeftIcon} boxSize="3.5" flexShrink={0} />
        <Text ref={labelRef} css={VIEW_MENU_LABEL_CSS} fontWeight="600" truncate>
          {text}
        </Text>
      </Button>
    </ClippedTooltip>
  );
};

const CenterViewMenu = ({
  activeItem,
  availableItems,
  isCloseDisabled,
  items,
  onClose,
  onOpen,
  onSelect,
}: {
  activeItem?: CenterWidgetItem;
  availableItems: WidgetEnableMenuItem[];
  isCloseDisabled: boolean;
  items: CenterWidgetItem[];
  /** Absent while the workspace is locked: closing a view removes it. */
  onClose?: () => void;
  onOpen: (item: WidgetEnableMenuItem) => void;
  onSelect: (instanceId: string) => void;
}) => {
  const { t } = useTranslation();
  const label = activeItem?.label ?? t('widgets.centerViewEmpty');
  const triggerLabel = t('widgets.centerViewLabel', { label });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const labelRef = useRef<HTMLParagraphElement>(null);
  const triggerIds = useTooltipTriggerIds();
  // The tooltip names the view only while the container query hides its name.
  const isLabelHidden = useCallback(() => labelRef.current?.getClientRects().length === 0, []);
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
    <Menu.Root ids={triggerIds} positioning={CENTER_MENU_POSITIONING}>
      <ClippedTooltip content={triggerLabel} ids={triggerIds} isClipped={isLabelHidden}>
        <Menu.Trigger asChild>
          <IconButton
            ref={triggerRef}
            aria-label={triggerLabel}
            color="fg"
            // The button recipe refuses to shrink; a narrow center truncates the view name beside the widget's label,
            // and a compact center hides it, leaving the icon and chevron.
            css={VIEW_MENU_TRIGGER_CSS}
            flexShrink={1}
            minW={VIEW_MENU_MIN_WIDTH}
            overflow="hidden"
            px="2"
            size="sm"
            variant="ghost"
          >
            <HStack css={VIEW_MENU_CONTENT_CSS} minW="0">
              {activeItem ? <CenterViewIcon instanceId={activeItem.id} widget={activeItem.widget} /> : null}
              <Text ref={labelRef} css={VIEW_MENU_LABEL_CSS} fontWeight="700" truncate>
                {label}
              </Text>
              <ChevronDownIcon size={12} />
            </HStack>
          </IconButton>
        </Menu.Trigger>
      </ClippedTooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="xs" textTransform="uppercase">
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
                  <Menu.ItemGroupLabel color="fg.subtle" fontSize="xs" textTransform="uppercase">
                    {t('widgets.centerViewsAdd')}
                  </Menu.ItemGroupLabel>
                  {availableItems.map((item) => (
                    <CenterViewMenuRow key={item.id} item={item} onOpen={onOpen} />
                  ))}
                </Menu.ItemGroup>
              </>
            ) : null}

            {activeItem && onClose ? (
              <>
                <Menu.Separator borderColor="border.subtle" />
                <Menu.Item disabled={isCloseDisabled} value="close-center-view" onClick={onClose}>
                  <Icon as={XIcon} boxSize="3.5" color="fg.subtle" />
                  <Menu.ItemText>{t('widgets.centerViewClose', { label: activeItem.label })}</Menu.ItemText>
                </Menu.Item>
              </>
            ) : null}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/** Suspense for the center icon: DeferredResource has no subscription and center widgets lack frame headers. */
const CenterViewIcon = ({ instanceId, widget }: { instanceId: string; widget: RegisteredWidget }) => {
  const fallback = useMemo(
    () => <WidgetIdentityIcon boxSize="3.5" icon={widget.manifest.icon} isLoading />,
    [widget.manifest.icon]
  );

  return (
    <Suspense fallback={fallback}>
      <WidgetChromeSlotById instanceId={instanceId} widget={widget} slot="icon" />
    </Suspense>
  );
};

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
      <Icon as={CheckIcon} boxSize="3" color="fg" opacity={isActive ? 1 : 0} />
      <WidgetIcon icon={item.icon} boxSize="3.5" />
      <Menu.ItemText>{item.label}</Menu.ItemText>
    </Menu.Item>
  );
};

/** Resolve kept instances from widgetInstances: presets replace region membership but merge and preserve instances. */
const KeptCenterViewSlot = ({ instanceId }: { instanceId: string }) => {
  const instance = useActiveProjectSelector(
    (project) => project.widgetInstances[instanceId],
    areWidgetRenderInstancesEqual
  );
  const widget = instance ? getWidgetById(instance.typeId) : undefined;

  if (!instance || !widget || widget.status !== 'enabled') {
    return <FallbackCenterView />;
  }

  return <WidgetRendererById instanceId={instance.id} widget={widget} region="center" />;
};

const FallbackCenterView = () => {
  const { t } = useTranslation();

  return (
    <Flex align="center" h="full" justify="center" w="full">
      <Text color="fg.subtle" fontSize="lg">
        {t('common.widgetViewUnavailable')}
      </Text>
    </Flex>
  );
};
