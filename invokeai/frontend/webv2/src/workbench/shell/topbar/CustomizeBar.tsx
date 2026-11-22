import type { LayoutPresetRoute } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { CustomizeSession } from '@workbench/workspace/structureLock';
import type { ChangeEvent, ReactNode } from 'react';

import { Box, HStack, Icon, Input, Popover, Portal, Separator, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { HIDDEN_SCROLLBAR_CSS } from '@platform/ui/scrollStripEdges';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { getPlacedWidgetTypeIds, graphWidgetSources } from '@workbench/graphWidgets';
import { useShortcutLabel } from '@workbench/hotkeys/useShortcutBinding';
import { getDestinationLabel } from '@workbench/invocation';
import { getLayoutPresetLabel, layoutPresets } from '@workbench/layoutPresets';
import {
  areLayoutPresetSnapshotsEqual,
  createLayoutPresetSnapshot,
  resolveSavedLayoutPreset,
} from '@workbench/layoutPresetSnapshots';
import { CUE_WIDTH, OverflowCue, useOverflowEdges } from '@workbench/shell/overflowEdges';
import { openWidgetsDialogFrom } from '@workbench/shell/widgetsDialogStore';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { getCustomizeSession, isCustomizeResetPending } from '@workbench/workspace/structureLock';
import {
  claimCustomizeBarFocus,
  customizeNewWorkspace,
  restoreFocusAfterCustomize,
} from '@workbench/workspace/workspaceNavigation';
import {
  ChevronDownIcon,
  CopyPlusIcon,
  LayoutGridIcon,
  PencilRulerIcon,
  Redo2Icon,
  RotateCcwIcon,
  Settings2Icon,
  Undo2Icon,
  WaypointsIcon,
} from 'lucide-react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { getInitialLayoutPresetIconId, resolveLayoutPresetRoute } from './layoutPresetDialogModel';
import { LayoutPresetIconPicker } from './LayoutPresetIconPicker';
import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { openLayoutPresetManager } from './layoutPresetManagerStore';
import { LayoutPresetRouteFields } from './LayoutPresetRouteFields';
import { HIDE_BELOW_PRESET_LABEL_WIDTH } from './topbarBreakpoints';

const POPOVER_POSITIONING = { placement: 'bottom' } as const;
const BAR_FOCUS_RING = { outline: '2px solid', outlineColor: 'accent.focusRing', outlineOffset: '1px' } as const;

const selectPlacedGraphWidgetSources = (project: Project) => {
  const placedTypeIds = getPlacedWidgetTypeIds(project);

  return graphWidgetSources.filter((source) => placedTypeIds.has(source.typeId));
};
const selectHistory = (project: Project) =>
  `${project.undoRedo.past.length > 0 ? 'u' : ''}${project.undoRedo.future.length > 0 ? 'r' : ''}`;

const isSameRoute = (left: LayoutPresetRoute | null, right: LayoutPresetRoute | null): boolean =>
  left?.sourceId === right?.sourceId && left?.destination === right?.destination;

/**
 * Replaces the workspace tabs while the active project customizes its workspace. Name, icon and route are edited here
 * and land with Done; the arrangement is edited live in the shell. Done saves it if this session changed it; Cancel
 * lays out what was on screen when Customize began.
 */
export const CustomizeBar = ({ session }: { session: CustomizeSession }) => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const { layout } = commands;
  const queries = useWorkbenchQueries();
  const { draft, entry, presetId } = session;
  const projectId = useActiveProjectSelector((project) => project.id);
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const sourceOptions = useActiveProjectSelector(selectPlacedGraphWidgetSources, areSourceListsEqual);
  const history = useActiveProjectSelector(selectHistory);
  const isResetPending = useActiveProjectSelector(isCustomizeResetPending);
  const undoShortcut = useShortcutLabel('app.undoLayoutChange');
  const redoShortcut = useShortcutLabel('app.redoLayoutChange');
  const widgetsButtonRef = useRef<HTMLButtonElement>(null);
  const { attach: attachScroller, hiddenEdges, scrollTo } = useOverflowEdges();

  const shipped = layoutPresets.find((preset) => preset.id === presetId);
  // A pending reset shows, and Done measures against, what the built-in ships with.
  const base = isResetPending && shipped ? shipped : resolveSavedLayoutPreset(account, presetId);
  const baseLabel = getLayoutPresetLabel(base, t);
  const allowsAuto = shipped?.defaultRoute === undefined;
  const name = draft.label ?? baseLabel;
  const iconId = getInitialLayoutPresetIconId(draft.iconId ?? base.iconId);
  const route = resolveLayoutPresetRoute(
    draft.defaultRoute === undefined ? base.defaultRoute : draft.defaultRoute,
    sourceOptions,
    allowsAuto
  );
  const isNamed = name.trim().length > 0;
  const hasOverride =
    shipped !== undefined &&
    !isResetPending &&
    [account.layoutPresetMetadataOverrides, account.layoutPresetOverrides, account.layoutPresetRouteOverrides].some(
      (overrides) => overrides?.[shipped.id] !== undefined
    );

  // Entering by a gesture moves focus here. However the session ends while this project is on screen, focus goes back;
  // switching projects unmounts the bar too, and that leaves focus alone.
  const bindBar = useCallback(
    (element: HTMLDivElement | null) => {
      if (!element) {
        return;
      }
      if (claimCustomizeBarFocus(projectId)) {
        element.focus({ preventScroll: true });
      }

      return () => {
        const { activeProject, customizing } = queries.getSnapshot();

        if (activeProject.id === projectId && getCustomizeSession(customizing, activeProject) === null) {
          restoreFocusAfterCustomize(projectId);
        }
      };
    },
    [projectId, queries]
  );
  const setName = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => layout.setCustomizeDraft({ label: event.currentTarget.value }),
    [layout]
  );
  const setIcon = useCallback((nextIconId: string) => layout.setCustomizeDraft({ iconId: nextIconId }), [layout]);
  const setRoute = useCallback(
    (nextRoute: LayoutPresetRoute | null) => layout.setCustomizeDraft({ defaultRoute: nextRoute }),
    [layout]
  );
  const undo = useCallback(() => layout.undo(), [layout]);
  const redo = useCallback(() => layout.redo(), [layout]);
  const openWidgets = useCallback(() => openWidgetsDialogFrom(() => widgetsButtonRef.current), []);
  const resetToDefault = useCallback(() => layout.resetCustomizeToDefault(), [layout]);
  const saveAsNew = useCallback(
    () =>
      customizeNewWorkspace(
        { commands, queries },
        {
          defaultRoute: route,
          iconId,
          label:
            name.trim() && name.trim() !== baseLabel ? name.trim() : t('topbar.presets.copyName', { name: baseLabel }),
        },
        { continues: true }
      ),
    [baseLabel, commands, iconId, name, queries, route, t]
  );
  const cancel = useCallback(() => layout.endCustomize(true), [layout]);
  const done = useCallback(() => {
    const label = name.trim();

    if (!label) {
      return;
    }

    if (isResetPending) {
      layout.restorePresetDefault(presetId);
    }
    if (draft.defaultRoute !== undefined && !isSameRoute(route, base.defaultRoute ?? null)) {
      layout.setPresetRoute(presetId, route);
    }
    if (label !== baseLabel) {
      layout.renamePreset(presetId, label, shipped ? getLayoutPresetLabel(shipped, t) : undefined);
    }
    if (draft.iconId !== undefined && draft.iconId !== base.iconId) {
      layout.setPresetIcon(presetId, draft.iconId);
    }

    const live = createLayoutPresetSnapshot(queries.getSnapshot().activeProject);

    // Only an arrangement this session changed is saved: saving one merely looked at differently would pin a built-in
    // to it instead of letting it follow the shipped one. After a reset, the saved arrangement is the shipped one.
    if (
      isResetPending ? !areLayoutPresetSnapshotsEqual(live, base.snapshot) : !areLayoutPresetSnapshotsEqual(live, entry)
    ) {
      layout.savePreset(presetId);
    }

    layout.endCustomize(false);
  }, [base, baseLabel, draft, entry, isResetPending, layout, name, presetId, queries, route, shipped, t]);

  return (
    <HStack
      ref={bindBar}
      aria-label={t('shell.workspace.customizing', { name: baseLabel })}
      bg="bg.subtle"
      borderColor="accent.solid"
      borderWidth="1px"
      data-customize-bar=""
      gap="1"
      h="9"
      maxW="min(72vw, 60rem)"
      minW="0"
      outline="none"
      pe="1"
      ps="2"
      role="group"
      rounded="lg"
      tabIndex={-1}
      _focusVisible={BAR_FOCUS_RING}
    >
      {/* A narrow window scrolls the editing controls sideways, with cues at the ends hiding some (the name field gives
          way first); Cancel and Done stay in view. */}
      <Box flex="0 1 auto" minW="0" position="relative">
        <HStack
          ref={attachScroller}
          css={HIDDEN_SCROLLBAR_CSS}
          gap="1"
          minW="0"
          overflowX="auto"
          overscrollBehaviorX="contain"
          py="0.5"
          scrollPaddingInline={CUE_WIDTH}
        >
          <HStack color="accent.fg" flexShrink={0} gap="1.5">
            <Icon as={PencilRulerIcon} boxSize="3.5" />
            <Text css={HIDE_BELOW_PRESET_LABEL_WIDTH} fontWeight="600">
              {t('shell.workspace.customize')}
            </Text>
          </HStack>
          <Separator flexShrink={0} h="5" mx="0.5" orientation="vertical" />
          <IconPopover icon={resolveLayoutPresetIcon(iconId)} label={t('topbar.presets.icon')}>
            <LayoutPresetIconPicker iconId={iconId} onChange={setIcon} />
          </IconPopover>
          <Input
            aria-invalid={!isNamed}
            aria-label={t('topbar.presets.name')}
            autoComplete="off"
            flex="0 1 11rem"
            minW="6rem"
            name="workspace-name"
            size="sm"
            value={name}
            onChange={setName}
          />
          <RoutePopover
            route={route}
            sourceLabel={sourceOptions.find((source) => source.sourceId === route?.sourceId)?.label}
          >
            <LayoutPresetRouteFields
              allowsAuto={allowsAuto}
              route={route}
              sourceOptions={sourceOptions}
              onChange={setRoute}
            />
          </RoutePopover>
          <Separator flexShrink={0} h="5" mx="0.5" orientation="vertical" />
          <IconButton
            aria-label={t('common.undo')}
            disabled={!history.includes('u')}
            size="sm"
            tooltip={undoShortcut ? `${t('common.undo')} (${undoShortcut})` : t('common.undo')}
            variant="ghost"
            onClick={undo}
          >
            <Icon as={Undo2Icon} boxSize="3.5" />
          </IconButton>
          <IconButton
            aria-label={t('common.redo')}
            disabled={!history.includes('r')}
            size="sm"
            tooltip={redoShortcut ? `${t('common.redo')} (${redoShortcut})` : t('common.redo')}
            variant="ghost"
            onClick={redo}
          >
            <Icon as={Redo2Icon} boxSize="3.5" />
          </IconButton>
          <Separator flexShrink={0} h="5" mx="0.5" orientation="vertical" />
          <IconButton
            ref={widgetsButtonRef}
            aria-label={t('shell.widgets.menuItem')}
            size="sm"
            tooltip={t('shell.widgets.menuItem')}
            variant="ghost"
            onClick={openWidgets}
          >
            <Icon as={LayoutGridIcon} boxSize="3.5" />
          </IconButton>
          {hasOverride ? (
            <IconButton
              aria-label={t('topbar.presets.resetToDefault')}
              size="sm"
              tooltip={t('topbar.presets.resetToDefault')}
              variant="ghost"
              onClick={resetToDefault}
            >
              <Icon as={RotateCcwIcon} boxSize="3.5" />
            </IconButton>
          ) : null}
          <IconButton
            aria-label={t('topbar.presets.saveAs')}
            size="sm"
            tooltip={t('topbar.presets.saveAs')}
            variant="ghost"
            onClick={saveAsNew}
          >
            <Icon as={CopyPlusIcon} boxSize="3.5" />
          </IconButton>
          <IconButton
            aria-label={t('topbar.presets.manage')}
            size="sm"
            tooltip={t('topbar.presets.manage')}
            variant="ghost"
            data-manage-workspaces=""
            onClick={openLayoutPresetManager}
          >
            <Icon as={Settings2Icon} boxSize="3.5" />
          </IconButton>
        </HStack>
        {hiddenEdges.start ? (
          <OverflowCue edge="start" label={t('common.scrollToStart')} onScrollTo={scrollTo} />
        ) : null}
        {hiddenEdges.end ? <OverflowCue edge="end" label={t('common.scrollToEnd')} onScrollTo={scrollTo} /> : null}
      </Box>
      <Separator flexShrink={0} h="5" mx="0.5" orientation="vertical" />
      <Button flexShrink={0} size="sm" variant="ghost" onClick={cancel}>
        {t('common.cancel')}
      </Button>
      <Button
        colorPalette="accent"
        flexShrink={0}
        size="sm"
        unavailableReason={isNamed ? undefined : t('topbar.presets.nameRequired')}
        onClick={done}
      >
        {t('common.done')}
      </Button>
    </HStack>
  );
};

const areSourceListsEqual = (left: readonly unknown[], right: readonly unknown[]): boolean =>
  left.length === right.length && left.every((source, index) => source === right[index]);

/** An icon-only popover trigger, named and tipped by its label. */
const IconPopover = ({
  children,
  icon,
  label,
}: {
  children: ReactNode;
  icon: ReturnType<typeof resolveLayoutPresetIcon>;
  label: string;
}) => {
  const triggerIds = useTooltipTriggerIds();

  return (
    <Popover.Root ids={triggerIds} lazyMount positioning={POPOVER_POSITIONING} unmountOnExit>
      <Tooltip content={label} ids={triggerIds}>
        <Popover.Trigger asChild>
          <IconButton aria-label={label} flexShrink={0} size="sm" variant="ghost">
            <Icon as={icon} boxSize="4" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent aria-label={label} p="3" w="auto">
            {children}
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

/** The route trigger reads the route itself: Auto, or "Source → Destination"; compact, it keeps a tip. */
const RoutePopover = ({
  children,
  route,
  sourceLabel,
}: {
  children: ReactNode;
  route: LayoutPresetRoute | null;
  sourceLabel: string | undefined;
}) => {
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();
  const summary =
    route && sourceLabel
      ? t('topbar.presets.routeSummary', { destination: getDestinationLabel(route.destination), source: sourceLabel })
      : t('topbar.presets.autoRoute');
  const label = t('topbar.presets.routeNamed', { route: summary });

  return (
    <Popover.Root ids={triggerIds} lazyMount positioning={POPOVER_POSITIONING} unmountOnExit>
      <Tooltip content={label} ids={triggerIds}>
        <Popover.Trigger asChild>
          <Button aria-label={label} flexShrink={1} gap="1.5" minW="0" px="2" size="sm" variant="ghost">
            <Icon as={WaypointsIcon} boxSize="3.5" flexShrink={0} />
            <Text css={HIDE_BELOW_PRESET_LABEL_WIDTH} truncate>
              {summary}
            </Text>
            <Icon as={ChevronDownIcon} boxSize="3" flexShrink={0} />
          </Button>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent aria-label={t('topbar.presets.route')} p="3" w="20rem">
            {children}
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
