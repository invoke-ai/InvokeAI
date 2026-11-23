import type { DragEndEvent } from '@dnd-kit/core';
import type { BuiltInLayoutPresetId, LayoutPreset, LayoutPresetId } from '@workbench/layoutContracts';
import type { KeyboardEvent, MouseEvent, PointerEvent } from 'react';

import { Box, HStack, Icon, Kbd, Menu, Portal, Stack, Text, useTooltipContext, VisuallyHidden } from '@chakra-ui/react';
import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { restrictToHorizontalAxis, restrictToParentElement } from '@dnd-kit/modifiers';
import {
  horizontalListSortingStrategy,
  sortableKeyboardCoordinates,
  SortableContext,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { IconButton } from '@platform/ui/Button';
import { MenuContent, MenuHeadingGroup, MenuItem } from '@platform/ui/Menu';
import { HIDDEN_SCROLLBAR_CSS } from '@platform/ui/scrollStripEdges';
import { Tabs } from '@platform/ui/Tabs';
import { Tooltip, TooltipGroup } from '@platform/ui/Tooltip';
import { useShortcutBinding, useShortcutLabel } from '@workbench/hotkeys/useShortcutBinding';
import { preloadLayoutWidgets } from '@workbench/layoutPresetActivation';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import {
  builtInLayoutPresetDescriptors,
  getLayoutPresetLabel,
  isBuiltInLayoutPresetId,
} from '@workbench/layoutPresets';
import { CUE_WIDTH, OverflowCue, useOverflowEdges } from '@workbench/shell/overflowEdges';
import {
  useActiveProjectSelector,
  useActiveWorkspaceId,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { useCanEditStructure } from '@workbench/workspace/useStructureLock';
import {
  customizeNewWorkspace,
  getUniqueWorkspaceName,
  getWorkspaceTab,
  startCustomizing,
} from '@workbench/workspace/workspaceNavigation';
import {
  ArrowRightIcon,
  ChevronDownIcon,
  PencilRulerIcon,
  PlusIcon,
  RotateCcwSquareIcon,
  SettingsIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { openLayoutPresetDelete, openLayoutPresetManager } from './layoutPresetManagerStore';
import { HIDE_BELOW_PRESET_LABEL_WIDTH } from './topbarBreakpoints';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';

const PRESET_MENU_ATTRIBUTE = 'data-preset-menu';
const TAB_FILL_TRANSITION =
  'background var(--chakra-durations-faster) ease, border-color var(--chakra-durations-faster) ease, color var(--chakra-durations-faster) ease';
const PRESET_TAB_KEYS_BLOCKED_DURING_DRAG = new Set(['ArrowDown', 'ArrowLeft', 'ArrowRight', 'End', 'Home']);
const DND_MODIFIERS = [restrictToHorizontalAxis, restrictToParentElement];
/** Counts the presets past the end, and keeps the active one in view when the strip narrows or the preset changes. */
const PRESET_OVERFLOW_OPTIONS = {
  itemSelector: '[role="tab"]',
  keepInViewSelector: '[role="tab"][aria-selected="true"]',
};
const KEYBOARD_SENSOR_OPTIONS = {
  coordinateGetter: sortableKeyboardCoordinates,
  keyboardCodes: { cancel: ['Escape'], end: ['Space', 'Tab'], start: ['Space'] },
};
const MOUSE_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;
const TOUCH_SENSOR_OPTIONS = { activationConstraint: { delay: 250, tolerance: 5 } } as const;

/** What each built-in preset is for, as the tab's tip says under its name. */
const BUILT_IN_PRESET_DESCRIPTION_KEYS: Record<BuiltInLayoutPresetId, string> = {
  automate: 'topbar.presets.builtInDescriptions.automate',
  compose: 'topbar.presets.builtInDescriptions.compose',
  edit: 'topbar.presets.builtInDescriptions.edit',
  video: 'topbar.presets.builtInDescriptions.video',
};

const findPresetShortcutCommandId = (presetId: LayoutPresetId): string | null => {
  const descriptor = builtInLayoutPresetDescriptors.find(({ preset }) => preset.id === presetId);

  return descriptor ? `app.${descriptor.hotkeyId}` : null;
};

// The label's own hiding condition, so the tip and the label never disagree about whether the name is on screen.
const matchLabelHidden = (): MediaQueryList =>
  globalThis.matchMedia(Object.keys(HIDE_BELOW_PRESET_LABEL_WIDTH)[0]!.replace(/^@media\s+/, ''));
const subscribeLabelHidden = (onChange: () => void): (() => void) => {
  const query = matchLabelHidden();
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
};
const getLabelHidden = (): boolean => matchLabelHidden().matches;

const NEW_WORKSPACE_ATTRIBUTE = 'data-new-workspace';
const getNewWorkspaceButton = (): HTMLElement | null => document.querySelector(`[${NEW_WORKSPACE_ATTRIBUTE}]`);

export const LayoutPresetStrip = () => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const { layout } = commands;
  // Reordering workspaces is structure; with the tabs locked a drag does nothing.
  const canEditStructure = useCanEditStructure();
  const [menuTarget, setMenuTarget] = useState<PresetMenuTarget | null>(null);
  const menuTicket = useRef(0);

  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  // Labels as shown: a built-in preset not renamed by the account reads in the current language.
  const presets = useMemo(
    () => getOrderedLayoutPresets(account).map((preset) => ({ ...preset, label: getLayoutPresetLabel(preset, t) })),
    [account, t]
  );
  const presetIds = useMemo(() => presets.map(({ id }) => id), [presets]);
  const activePresetId = useActiveWorkspaceId();
  const [isReordering, setIsReordering] = useState(false);
  const isLabelHidden = useSyncExternalStore(subscribeLabelHidden, getLabelHidden);
  // Like the top bar's other labelled controls, a visible label is not repeated: beside it, a preset has a tip only for
  // what the label does not say (a built-in's shortcut and purpose).
  const hasTip = useCallback(
    (presetId: LayoutPresetId) => isLabelHidden || isBuiltInLayoutPresetId(presetId),
    [isLabelHidden]
  );
  // Quiet while a preset's menu is open or a tab is being dragged, where it would cover the menu or trail the drag.
  const isTipShown = useCallback(
    (presetId: LayoutPresetId) => menuTarget === null && !isReordering && hasTip(presetId),
    [hasTip, isReordering, menuTarget]
  );
  const presetTip = useMemo(() => <PresetTipContent hasTip={hasTip} presets={presets} />, [hasTip, presets]);
  const tabsIdPrefix = useId();
  // Named so the strip's tooltip can anchor to each tab's own id instead of replacing it.
  const tabIds = useMemo(() => ({ trigger: (value: string) => `${tabsIdPrefix}-preset-${value}` }), [tabsIdPrefix]);
  const [pendingPresetId, setPendingPresetId] = useState<LayoutPresetId | null>(null);

  // The store caught up; stop overriding it.
  if (pendingPresetId !== null && pendingPresetId === activePresetId) {
    setPendingPresetId(null);
  }

  const selectedPresetId = pendingPresetId ?? activePresetId;
  const activePreset = useMemo(
    () => presets.find(({ id }) => id === selectedPresetId) ?? presets[0]!,
    [presets, selectedPresetId]
  );
  const dndAccessibility = useMemo(
    () => ({ screenReaderInstructions: { draggable: t('topbar.presets.reorderInstructions') } }),
    [t]
  );
  // Presets past the strip's width say so at its ends, like the center's chrome rows.
  const { attach: attachScroller, hiddenEdges, scrollTo } = useOverflowEdges(PRESET_OVERFLOW_OPTIONS);
  const sensors = useSensors(
    useSensor(MouseSensor, MOUSE_SENSOR_OPTIONS),
    useSensor(TouchSensor, TOUCH_SENSOR_OPTIONS),
    useSensor(KeyboardSensor, KEYBOARD_SENSOR_OPTIONS)
  );
  // The latest tab press; a frame-deferred activation that a newer press superseded does not start.
  const requestedPresetIdRef = useRef<LayoutPresetId | null>(null);
  /** Acknowledge the tab press before rearranging the workspace so a blocking layout commit cannot delay feedback. */
  const requestPreset = useCallback(
    (presetId: LayoutPresetId) => {
      // Pressing the preset the workbench is still on, while another is pending, means stay: cancel the pending switch.
      if (presetId === activePresetId) {
        requestedPresetIdRef.current = null;
        layout.cancelPresetActivation();
        setPendingPresetId(null);
        return;
      }

      requestedPresetIdRef.current = presetId;
      setPendingPresetId(presetId);
      requestAnimationFrame(() => {
        if (requestedPresetIdRef.current !== presetId) {
          return;
        }

        void layout.activatePreset(presetId).then((appliedPresetId) => {
          // Return dropped activations to store selection without overwriting a newer request's optimistic selection.
          if (appliedPresetId !== presetId) {
            setPendingPresetId((pending) => (pending === presetId ? null : pending));
          }
        });
      });
    },
    [activePresetId, layout]
  );
  /** Pressing the active tab goes home, unless that press only cancelled a pending switch. */
  const goHome = useCallback(
    (presetId: LayoutPresetId) => {
      if (presetId === activePresetId && pendingPresetId === null) {
        layout.applyPreset(presetId);
      }
    },
    [activePresetId, layout, pendingPresetId]
  );
  const applyPreset = useCallback(
    (preset: LayoutPreset) => {
      if (preset.id !== selectedPresetId) {
        requestPreset(preset.id);
      }
    },
    [requestPreset, selectedPresetId]
  );
  const handleValueChange = useCallback(
    (event: { value: string }) => {
      const preset = presets.find((candidate) => candidate.id === event.value);

      if (preset) {
        applyPreset(preset);
      }
    },
    [applyPreset, presets]
  );
  // Stays available while locked: it is how a new place gets made, and Customize drafts it from what is on screen.
  const createWorkspace = useCallback(
    () =>
      customizeNewWorkspace(
        { commands, queries },
        {
          defaultRoute: activePreset.defaultRoute ?? null,
          iconId: activePreset.iconId,
          label: getUniqueWorkspaceName(
            t('topbar.presets.copyName', { name: activePreset.label }),
            presets.map((preset) => preset.label)
          ),
        },
        { returnFocus: getNewWorkspaceButton }
      ),
    [activePreset, commands, presets, queries, t]
  );
  const customize = useCallback(
    (preset: LayoutPreset) =>
      startCustomizing({ commands, queries }, { returnFocus: () => getWorkspaceTab(preset.id) }),
    [commands, queries]
  );
  const openMenu = useCallback((target: Omit<PresetMenuTarget, 'ticket'>) => {
    menuTicket.current += 1;
    setMenuTarget({ ...target, ticket: menuTicket.current });
  }, []);
  const closeMenu = useCallback(() => setMenuTarget(null), []);
  const requestDelete = useCallback((preset: LayoutPreset) => openLayoutPresetDelete(preset.id), []);
  const handleDragStart = useCallback(() => setIsReordering(true), []);
  const handleDragCancel = useCallback(() => setIsReordering(false), []);
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      setIsReordering(false);

      if (!event.over || event.active.id === event.over.id) {
        return;
      }

      layout.reorderPresets(event.active.id as LayoutPresetId, event.over.id as LayoutPresetId);
    },
    [layout]
  );

  return (
    <>
      <HStack gap="1" justify="center" maxW="min(36vw, 44rem)" minW="0">
        {/* The cues' positioning context; the scroller fills it. */}
        <Box maxW="full" minW="0" position="relative">
          <Box
            ref={attachScroller}
            css={HIDDEN_SCROLLBAR_CSS}
            data-layout-preset-scroll=""
            isolation="isolate"
            maxW="full"
            minW="0"
            overflowX="auto"
            overscrollBehaviorX="contain"
            // Focus and keep-in-view scroll a tab clear of the cues, not merely onto the strip's edge.
            scrollPaddingInline={CUE_WIDTH}
          >
            <DndContext
              accessibility={dndAccessibility}
              autoScroll={false}
              collisionDetection={closestCenter}
              modifiers={DND_MODIFIERS}
              sensors={sensors}
              onDragCancel={handleDragCancel}
              onDragEnd={handleDragEnd}
              onDragStart={handleDragStart}
            >
              <Tabs.Root
                ids={tabIds}
                minW="max-content"
                size="lg"
                value={selectedPresetId}
                variant="subtle"
                onValueChange={handleValueChange}
              >
                {/* The name belongs on the tablist, not the root — the root is a plain
                  container and carries no role for it to name. */}
                <SortableContext items={presetIds} strategy={horizontalListSortingStrategy}>
                  {/* One tip for the strip: it follows focus and the pointer from tab to tab. Not closed by scrolling:
                    an arrow-key switch re-lays out the workbench, whose panels restore their scroll positions, and the
                    tab itself never scrolls out from under the tip. */}
                  <TooltipGroup.Root
                    closeOnScroll={false}
                    content={presetTip}
                    getTriggerId={tabIds.trigger}
                    isEnabled={isTipShown}
                  >
                    <Tabs.List aria-label={t('topbar.presets.workspaces')} gap="0.5">
                      {presets.map((preset) => (
                        <PresetTab
                          key={preset.id}
                          canReorder={canEditStructure}
                          isActive={preset.id === selectedPresetId}
                          preset={preset}
                          onGoHome={goHome}
                          onOpenMenu={openMenu}
                          onRequest={requestPreset}
                        />
                      ))}
                    </Tabs.List>
                  </TooltipGroup.Root>
                </SortableContext>
                {/* Provide aria-controls targets describing the shared dock outside this component. Pinned to the
                    row's start: at its place in the flow, the shown panel would lie past the tabs and give the row
                    overflow that hides nothing. */}
                {presets.map((preset) => (
                  <Tabs.Content key={preset.id} value={preset.id} asChild>
                    <VisuallyHidden insetInlineStart="0" top="0">
                      {t('topbar.presets.workspaceNamed', { name: preset.label })}
                    </VisuallyHidden>
                  </Tabs.Content>
                ))}
              </Tabs.Root>
            </DndContext>
          </Box>
          {hiddenEdges.start ? (
            <OverflowCue edge="start" label={t('common.scrollToStart')} onScrollTo={scrollTo} />
          ) : null}
          {hiddenEdges.end ? (
            <OverflowCue
              ariaLabel={t('topbar.presets.moreAvailable', { count: hiddenEdges.endCount })}
              count={hiddenEdges.endCount}
              edge="end"
              label={t('topbar.presets.more')}
              onScrollTo={scrollTo}
            />
          ) : null}
        </Box>

        <Tooltip content={t('topbar.presets.newWorkspace')} showArrow>
          <IconButton
            {...{ [NEW_WORKSPACE_ATTRIBUTE]: '' }}
            aria-label={t('topbar.presets.newWorkspace')}
            size="lg"
            variant="ghost"
            onClick={createWorkspace}
          >
            <Icon as={PlusIcon} boxSize="4" />
          </IconButton>
        </Tooltip>
      </HStack>

      {menuTarget ? (
        <PresetMenu
          key={menuTarget.ticket}
          canEditStructure={canEditStructure}
          isActive={menuTarget.preset.id === selectedPresetId}
          target={menuTarget}
          onApply={applyPreset}
          onClose={closeMenu}
          onCustomize={customize}
          onDelete={requestDelete}
        />
      ) : null}
    </>
  );
};

const PresetTab = ({
  canReorder,
  isActive,
  onGoHome,
  onOpenMenu,
  onRequest,
  preset,
}: {
  canReorder: boolean;
  isActive: boolean;
  preset: LayoutPreset;
  onGoHome: (presetId: LayoutPresetId) => void;
  onOpenMenu: (target: Omit<PresetMenuTarget, 'ticket'>) => void;
  onRequest: (presetId: LayoutPresetId) => void;
}) => {
  const icon = resolveLayoutPresetIcon(preset.iconId);
  // An unknown command id resolves to no binding, so a custom preset advertises no key.
  const shortcut = useShortcutBinding(findPresetShortcutCommandId(preset.id) ?? '');
  const keyShortcuts = [isActive ? 'ArrowDown' : null, shortcut?.aria].filter(Boolean).join(' ') || undefined;
  const sortable = useSortable({
    attributes: { role: 'tab', roleDescription: 'sortable', tabIndex: isActive ? 0 : -1 },
    disabled: !canReorder,
    id: preset.id,
  });
  const { isDragging, listeners, setNodeRef, transform, transition } = sortable;
  // Locked, a tab is a plain tab: not announced as sortable, nor marked disabled.
  const attributes = canReorder ? sortable.attributes : { role: 'tab', tabIndex: isActive ? 0 : -1 };
  const dndStyle = useMemo(
    () => ({
      opacity: isDragging ? 0.5 : undefined,
      position: 'relative' as const,
      transform: CSS.Translate.toString(transform),
      // Compose fill transitions with dnd-kit's inline transform transition so hover/selection fades remain active.
      transition: [transition, TAB_FILL_TRANSITION].filter(Boolean).join(', '),
      zIndex: isDragging ? 1 : undefined,
    }),
    [isDragging, transform, transition]
  );
  const queries = useWorkbenchQueries();
  // Warms what switching shows: the workspace as this project looks at it.
  const handlePreload = useCallback(
    () => preloadLayoutWidgets(queries.getWorkspaceLayout(preset.id)),
    [preset.id, queries]
  );

  // Whether the press began on the tab already active: a press that switched to it is not also a press to go home.
  const pressStartedActiveRef = useRef(false);

  // Use a span chevron recognized by the tab handler; nested buttons are invalid.
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const trigger = event.target instanceof Element ? event.target.closest(`[${PRESET_MENU_ATTRIBUTE}]`) : null;

      if (trigger) {
        event.preventDefault();
        event.stopPropagation();
        onOpenMenu({ anchor: trigger.getBoundingClientRect(), preset });
      } else if (event.detail === 0 ? isActive : pressStartedActiveRef.current) {
        onGoHome(preset.id);
      }
    },
    [isActive, onGoHome, onOpenMenu, preset]
  );

  // Activate on primary press for immediate feedback; right-click must preserve the inactive preset's menu actions.
  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      // Chain dnd-kit's activator so a sensor change cannot silently lose drag initiation.
      (listeners as { onPointerDown?: (value: unknown) => void } | undefined)?.onPointerDown?.(event);
      pressStartedActiveRef.current = isActive;

      if (event.button === 0 && event.pointerType !== 'touch' && !isActive) {
        onRequest(preset.id);
      }
    },
    [isActive, listeners, onRequest, preset.id]
  );

  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      onOpenMenu({
        anchor: new DOMRect(event.clientX, event.clientY, 1, 1),
        preset,
      });
    },
    [onOpenMenu, preset]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (isDragging && PRESET_TAB_KEYS_BLOCKED_DURING_DRAG.has(event.key)) {
        event.preventDefault();
      }

      listeners?.onKeyDown?.(event);

      if (event.defaultPrevented) {
        return;
      }

      if (!isActive || event.key !== 'ArrowDown') {
        return;
      }

      event.preventDefault();
      onOpenMenu({ anchor: event.currentTarget.getBoundingClientRect(), preset });
    },
    [isActive, isDragging, listeners, onOpenMenu, preset]
  );

  return (
    <TooltipGroup.Trigger value={preset.id}>
      <Tabs.Trigger
        ref={setNodeRef}
        {...attributes}
        {...listeners}
        aria-label={preset.label}
        aria-keyshortcuts={keyShortcuts}
        css={PRESET_TAB_CSS}
        cursor={isDragging ? 'grabbing' : 'default'}
        data-layout-preset-id={preset.id}
        gap="1.5"
        style={dndStyle}
        touchAction="pan-x"
        value={preset.id}
        _hover={PRESET_TAB_HOVER_PROPS}
        _selected={PRESET_TAB_SELECTED_PROPS}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onFocus={handlePreload}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointerDown}
        onPointerEnter={handlePreload}
      >
        <Box as="span" display="inline-flex" flexShrink={0} position="relative">
          <Icon as={icon} boxSize="3.5" color={isActive ? 'brand.fg' : undefined} />
        </Box>
        <Text as="span" css={HIDE_BELOW_PRESET_LABEL_WIDTH}>
          {preset.label}
        </Text>
        {isActive ? (
          <Box
            {...{ [PRESET_MENU_ATTRIBUTE]: '' }}
            alignItems="center"
            aria-hidden="true"
            as="span"
            color="fg.subtle"
            display="inline-flex"
            me="-1"
            p="0.5"
            rounded="sm"
            _hover={MENU_AFFORDANCE_HOVER_PROPS}
          >
            <Icon as={ChevronDownIcon} boxSize="3.5" />
          </Box>
        ) : null}
      </Tabs.Trigger>
    </TooltipGroup.Trigger>
  );
};

const MENU_AFFORDANCE_HOVER_PROPS = { bg: 'bg.hover', color: 'fg' } as const;

// A pointed inactive preset takes the shared tinted hover; the active one keeps the widget rail's active fill.
const PRESET_TAB_SURFACE = '--preset-tab-surface';
const PRESET_TAB_CSS = { [PRESET_TAB_SURFACE]: '{colors.bg.subtle}' } as const;
const PRESET_TAB_HOVER_PROPS = {
  '&:not([data-selected])': { bg: 'bg.hover', color: 'fg', [PRESET_TAB_SURFACE]: '{colors.bg.hover}' },
} as const;
const PRESET_TAB_SELECTED_PROPS = {
  bg: 'bg.emphasized',
  color: 'fg',
  [PRESET_TAB_SURFACE]: '{colors.bg.emphasized}',
} as const;

/** The strip's single tip for the tab it is on: name, shortcut and built-in purpose. */
const PresetTipContent = ({
  hasTip,
  presets,
}: {
  hasTip: (presetId: LayoutPresetId) => boolean;
  presets: readonly LayoutPreset[];
}) => {
  const { t } = useTranslation();
  const { triggerValue } = useTooltipContext();
  const preset = presets.find(({ id }) => id === triggerValue);

  // Focus moving off an open tip switches the tooltip's tab at once and closes the tip a render later: a tab with no
  // tip of its own must not show in the fading one.
  if (!preset || !hasTip(preset.id)) {
    return null;
  }

  const shortcutCommandId = findPresetShortcutCommandId(preset.id);

  return (
    <Stack gap="0.5" maxW="16rem">
      <HStack gap="1.5" justify="space-between">
        <Text as="span" fontWeight="600">
          {preset.label}
        </Text>
        {shortcutCommandId ? <PresetShortcutKeys commandId={shortcutCommandId} /> : null}
      </HStack>
      {/* A renamed built-in may have been turned to another purpose; its shipped description could contradict it. */}
      {isBuiltInLayoutPresetId(preset.id) && preset.labelKey !== undefined ? (
        <Text as="span" color="fg.muted">
          {t(BUILT_IN_PRESET_DESCRIPTION_KEYS[preset.id])}
        </Text>
      ) : null}
    </Stack>
  );
};

/** Reads the effective binding, so a remapped or unbound shortcut is never advertised. */
const PresetShortcutKeys = ({ commandId }: { commandId: string }) => {
  const binding = useShortcutBinding(commandId);

  return binding ? (
    <Kbd variant="subtle">
      <TopbarShortcutKeys parts={binding.parts} />
    </Kbd>
  ) : null;
};

interface PresetMenuTarget {
  anchor: DOMRect;
  preset: LayoutPreset;
  /** Remounts the menu per open so a repeated anchor still gets a fresh machine. */
  ticket: number;
}

/** Mounted only while open. Structural items show only while the workspace may change; otherwise Customize leads. */
const PresetMenu = ({
  canEditStructure,
  isActive,
  onApply,
  onClose,
  onCustomize,
  onDelete,
  target,
}: {
  canEditStructure: boolean;
  isActive: boolean;
  target: PresetMenuTarget;
  onApply: (preset: LayoutPreset) => void;
  onClose: () => void;
  onCustomize: (preset: LayoutPreset) => void;
  onDelete: (preset: LayoutPreset) => void;
}) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const customizeShortcut = useShortcutLabel('app.customizeWorkspace');
  const { preset } = target;
  const isCustom = preset.isBuiltIn !== true;
  const hasView = useActiveProjectSelector((project) => project.views[preset.id] !== undefined);

  const apply = useCallback(() => onApply(preset), [onApply, preset]);
  const customize = useCallback(() => {
    onClose();
    onCustomize(preset);
  }, [onClose, onCustomize, preset]);
  const remove = useCallback(() => {
    onClose();
    onDelete(preset);
  }, [onClose, onDelete, preset]);
  const resetView = useCallback(() => {
    onClose();
    layout.resetWorkspaceView(preset.id);
  }, [layout, onClose, preset.id]);
  const manage = useCallback(() => {
    onClose();
    openLayoutPresetManager();
  }, [onClose]);
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  // Anchored to a measured rect rather than a trigger element: the chevron lives inside the tab button and cannot be a
  // `Menu.Trigger` of its own without nesting buttons, and a right-click anchors at the pointer.
  const { anchor } = target;
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => ({ height: anchor.height, width: anchor.width, x: anchor.x, y: anchor.y }),
      placement: 'bottom-end' as const,
    }),
    [anchor]
  );

  return (
    <Menu.Root open positioning={positioning} onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="16rem">
            <MenuHeadingGroup label={preset.label}>
              {isActive ? (
                <Menu.Item value="customize" onClick={customize}>
                  <Icon as={PencilRulerIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('shell.workspace.customize')}</Menu.ItemText>
                  {customizeShortcut ? (
                    <Text color="fg.subtle" fontSize="xs" ms="auto">
                      {customizeShortcut}
                    </Text>
                  ) : null}
                </Menu.Item>
              ) : (
                <Menu.Item value="apply-preset" onClick={apply}>
                  <Icon as={ArrowRightIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('topbar.presets.switch')}</Menu.ItemText>
                </Menu.Item>
              )}

              <MenuItem
                unavailableReason={hasView ? undefined : t('topbar.presets.resetViewUnavailable')}
                value="reset-view"
                onClick={resetView}
              >
                <Icon as={RotateCcwSquareIcon} boxSize="3.5" />
                <Menu.ItemText>{t('topbar.presets.resetView')}</Menu.ItemText>
              </MenuItem>

              {canEditStructure && isCustom ? (
                <Menu.Item data-danger="" value="delete-preset" onClick={remove}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText>{t('topbar.presets.deleteWithEllipsis')}</Menu.ItemText>
                </Menu.Item>
              ) : null}
            </MenuHeadingGroup>

            {/* Like Customize, an explicit way into editing workspaces, so the lock does not hide it. */}
            <Menu.Separator />
            <Menu.Item value="manage-presets" onClick={manage}>
              <Icon as={SettingsIcon} boxSize="3.5" />
              <Menu.ItemText>{t('topbar.presets.manage')}</Menu.ItemText>
            </Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
