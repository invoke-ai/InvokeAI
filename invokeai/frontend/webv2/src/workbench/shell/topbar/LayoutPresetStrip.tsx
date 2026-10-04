import type { DragEndEvent } from '@dnd-kit/core';
import type { LayoutPreset, LayoutPresetId } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { KeyboardEvent, MouseEvent, PointerEvent } from 'react';

import { Box, HStack, Icon, Menu, Portal, Text, VisuallyHidden } from '@chakra-ui/react';
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
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tabs } from '@platform/ui/Tabs';
import { Tooltip } from '@platform/ui/Tooltip';
import { getPlacedWidgetTypeIds, graphWidgetSources } from '@workbench/graphWidgets';
import { preloadLayoutPresetWidgets } from '@workbench/layoutPresetActivation';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import {
  ArrowRightIcon,
  ChevronDownIcon,
  PencilIcon,
  PlusIcon,
  RotateCcwIcon,
  SaveIcon,
  SettingsIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayoutPresetDialogValue } from './layoutPresetDialogModel';

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { openLayoutPresetDelete, openLayoutPresetEdit, openLayoutPresetManager } from './layoutPresetManagerStore';
import { HIDE_BELOW_PRESET_LABEL_WIDTH } from './topbarBreakpoints';
import { useActiveLayoutPresetId, useLayoutDrift } from './useLayoutDrift';
import { useTopbarShortcut } from './useTopbarShortcut';

const PRESET_MENU_ATTRIBUTE = 'data-preset-menu';
const TAB_FILL_TRANSITION =
  'background var(--chakra-durations-faster) ease, border-color var(--chakra-durations-faster) ease, color var(--chakra-durations-faster) ease';
const PRESET_SCROLL_CSS = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' } as const;
const PRESET_TAB_KEYS_BLOCKED_DURING_DRAG = new Set(['ArrowDown', 'ArrowLeft', 'ArrowRight', 'End', 'Home']);
const DND_MODIFIERS = [restrictToHorizontalAxis, restrictToParentElement];
const KEYBOARD_SENSOR_OPTIONS = {
  coordinateGetter: sortableKeyboardCoordinates,
  keyboardCodes: { cancel: ['Escape'], end: ['Space', 'Tab'], start: ['Space'] },
};
const MOUSE_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;
const TOUCH_SENSOR_OPTIONS = { activationConstraint: { delay: 250, tolerance: 5 } } as const;

const createCustomPresetId = (): string =>
  `custom-layout-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const selectPlacedGraphWidgetSources = (project: Project) => {
  const placedTypeIds = getPlacedWidgetTypeIds(project);

  return graphWidgetSources.filter((source) => placedTypeIds.has(source.typeId));
};

export const LayoutPresetStrip = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const [isSaveAsOpen, setIsSaveAsOpen] = useState(false);
  const saveAsDialog = useExitPresence(isSaveAsOpen);
  const [menuTarget, setMenuTarget] = useState<PresetMenuTarget | null>(null);
  const menuTicket = useRef(0);

  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const invocation = useActiveProjectSelector((project) => project.invocation);
  const sourceOptions = useActiveProjectSelector(selectPlacedGraphWidgetSources);
  const presets = useMemo(() => getOrderedLayoutPresets(account), [account]);
  const presetIds = useMemo(() => presets.map(({ id }) => id), [presets]);
  const { hasDrifted } = useLayoutDrift();
  const activePresetId = useActiveLayoutPresetId();
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
  const sensors = useSensors(
    useSensor(MouseSensor, MOUSE_SENSOR_OPTIONS),
    useSensor(TouchSensor, TOUCH_SENSOR_OPTIONS),
    useSensor(KeyboardSensor, KEYBOARD_SENSOR_OPTIONS)
  );
  const saveAsDefaultRoute = useMemo(
    () => ({ destination: invocation.destination, sourceId: invocation.sourceId }),
    [invocation.destination, invocation.sourceId]
  );
  /** Acknowledge the tab press before rearranging the workspace so a blocking layout commit cannot delay feedback. */
  const requestPreset = useCallback(
    (presetId: LayoutPresetId) => {
      setPendingPresetId(presetId);
      requestAnimationFrame(() => {
        void layout.activatePreset(presetId).then((appliedPresetId) => {
          // Return dropped activations to store selection without overwriting a newer request's optimistic
          // selection.
          if (appliedPresetId !== presetId) {
            setPendingPresetId((pending) => (pending === presetId ? null : pending));
          }
        });
      });
    },
    [layout]
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
  const openSaveAsDialog = useCallback(() => setIsSaveAsOpen(true), []);
  const closeSaveAsDialog = useCallback(() => setIsSaveAsOpen(false), []);
  const saveAsNewPreset = useCallback(
    ({ defaultRoute, iconId, name }: LayoutPresetDialogValue) =>
      layout.createPreset(createCustomPresetId(), name, iconId, defaultRoute),
    [layout]
  );
  const openMenu = useCallback((target: Omit<PresetMenuTarget, 'ticket'>) => {
    menuTicket.current += 1;
    setMenuTarget({ ...target, ticket: menuTicket.current });
  }, []);
  const closeMenu = useCallback(() => setMenuTarget(null), []);
  const requestEdit = useCallback((preset: LayoutPreset) => openLayoutPresetEdit(preset.id), []);
  const requestDelete = useCallback((preset: LayoutPreset) => openLayoutPresetDelete(preset.id), []);
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
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
        <Box css={PRESET_SCROLL_CSS} data-layout-preset-scroll="" maxW="full" minW="0" overflowX="auto">
          <DndContext
            accessibility={dndAccessibility}
            autoScroll={false}
            collisionDetection={closestCenter}
            modifiers={DND_MODIFIERS}
            sensors={sensors}
            onDragEnd={handleDragEnd}
          >
            <Tabs.Root
              minW="max-content"
              size="lg"
              value={selectedPresetId}
              variant="subtle"
              onValueChange={handleValueChange}
            >
              {/* The name belongs on the tablist, not the root — the root is a plain
                  container and carries no role for it to name. */}
              <SortableContext items={presetIds} strategy={horizontalListSortingStrategy}>
                <Tabs.List aria-label={t('topbar.presets.layoutPreset')} gap="0.5">
                  {presets.map((preset) => (
                    <PresetTab
                      key={preset.id}
                      hasDrifted={hasDrifted}
                      isActive={preset.id === selectedPresetId}
                      preset={preset}
                      onOpenMenu={openMenu}
                      onRequest={requestPreset}
                    />
                  ))}
                </Tabs.List>
              </SortableContext>
              {/* Provide aria-controls targets describing the shared dock outside this component. */}
              {presets.map((preset) => (
                <Tabs.Content key={preset.id} value={preset.id} asChild>
                  <VisuallyHidden>{`${preset.label} layout`}</VisuallyHidden>
                </Tabs.Content>
              ))}
            </Tabs.Root>
          </DndContext>
        </Box>

        <Tooltip content={t('topbar.presets.saveAsTooltip')} showArrow>
          <IconButton
            aria-label={t('topbar.presets.saveAsTooltip')}
            size="lg"
            variant="ghost"
            onClick={openSaveAsDialog}
          >
            <Icon as={PlusIcon} boxSize="4" />
          </IconButton>
        </Tooltip>
      </HStack>

      {menuTarget ? (
        <PresetMenu
          key={menuTarget.ticket}
          isActive={menuTarget.preset.id === selectedPresetId}
          hasDrifted={hasDrifted}
          target={menuTarget}
          onApply={applyPreset}
          onClose={closeMenu}
          onDelete={requestDelete}
          onEdit={requestEdit}
        />
      ) : null}

      {saveAsDialog.isMounted ? (
        <LayoutPresetDialog
          key={saveAsDialog.generation}
          defaultRoute={saveAsDefaultRoute}
          isOpen={saveAsDialog.isOpen}
          name={`${activePreset.label} copy`}
          sourceOptions={sourceOptions}
          submitLabel={t('topbar.presets.save')}
          title={t('topbar.presets.saveAs')}
          onClose={closeSaveAsDialog}
          onExitComplete={saveAsDialog.release}
          onSubmit={saveAsNewPreset}
        />
      ) : null}
    </>
  );
};

const PresetTab = ({
  hasDrifted,
  isActive,
  onOpenMenu,
  onRequest,
  preset,
}: {
  hasDrifted: boolean;
  isActive: boolean;
  preset: LayoutPreset;
  onOpenMenu: (target: Omit<PresetMenuTarget, 'ticket'>) => void;
  onRequest: (presetId: LayoutPresetId) => void;
}) => {
  const { t } = useTranslation();
  const icon = resolveLayoutPresetIcon(preset.iconId);
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    attributes: { role: 'tab', roleDescription: 'sortable', tabIndex: isActive ? 0 : -1 },
    id: preset.id,
  });
  const dndStyle = useMemo(
    () => ({
      opacity: isDragging ? 0.5 : undefined,
      position: 'relative' as const,
      transform: CSS.Translate.toString(transform),
      // Compose fill transitions with dnd-kit's inline transform transition so hover/selection fades remain
      // active.
      transition: [transition, TAB_FILL_TRANSITION].filter(Boolean).join(', '),
      zIndex: isDragging ? 1 : undefined,
    }),
    [isDragging, transform, transition]
  );
  const handlePreload = useCallback(() => preloadLayoutPresetWidgets(preset), [preset]);
  const showDrift = isActive && hasDrifted;

  // Use a span chevron recognized by the tab handler; nested buttons are invalid.
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const trigger = event.target instanceof Element ? event.target.closest(`[${PRESET_MENU_ATTRIBUTE}]`) : null;

      if (trigger) {
        event.preventDefault();
        event.stopPropagation();
        onOpenMenu({ anchor: trigger.getBoundingClientRect(), preset });
      }
    },
    [onOpenMenu, preset]
  );

  // Activate on primary press for immediate feedback; right-click must preserve the inactive preset's menu
  // actions.
  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      // Chain dnd-kit's activator so a sensor change cannot silently lose drag initiation.
      (listeners as { onPointerDown?: (value: unknown) => void } | undefined)?.onPointerDown?.(event);

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
    <Tabs.Trigger
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      aria-label={showDrift ? `${preset.label}, ${t('topbar.presets.unsaved')}` : preset.label}
      aria-keyshortcuts={isActive ? 'ArrowDown' : undefined}
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
      <Icon as={icon} boxSize="3.5" color={isActive ? 'brand.fg' : undefined} flexShrink={0} />
      <Text as="span" css={HIDE_BELOW_PRESET_LABEL_WIDTH}>
        {preset.label}
      </Text>
      {showDrift ? <DriftDot /> : null}
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
  );
};

const MENU_AFFORDANCE_HOVER_PROPS = { bg: 'bg.hover', color: 'fg' } as const;

// A pointed inactive preset takes the shared tinted hover; the active one keeps the widget rail's active fill.
const PRESET_TAB_HOVER_PROPS = { '&:not([data-selected])': { bg: 'bg.hover', color: 'fg' } } as const;
const PRESET_TAB_SELECTED_PROPS = { bg: 'bg.emphasized', color: 'fg' } as const;

const DriftDot = () => <Box aria-hidden="true" bg="accent.solid" boxSize="1.5" flexShrink={0} rounded="full" />;

interface PresetMenuTarget {
  anchor: DOMRect;
  preset: LayoutPreset;
  /** Remounts the menu per open so a repeated anchor still gets a fresh machine. */
  ticket: number;
}

/**
 * Mounted only while open: closing unmounts it in the same commit that opens an admin dialog, so zag never sees the
 * dialog's layer nested above a menu that is still exiting (which dismisses the dialog).
 */
const PresetMenu = ({
  hasDrifted,
  isActive,
  onApply,
  onClose,
  onDelete,
  onEdit,
  target,
}: {
  hasDrifted: boolean;
  isActive: boolean;
  target: PresetMenuTarget;
  onApply: (preset: LayoutPreset) => void;
  onClose: () => void;
  onDelete: (preset: LayoutPreset) => void;
  onEdit: (preset: LayoutPreset) => void;
}) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const saveShortcut = useTopbarShortcut('app.saveLayoutPreset');
  const { preset } = target;
  const isCustom = preset.isBuiltIn !== true;
  const showDrift = isActive && hasDrifted;

  const apply = useCallback(() => onApply(preset), [onApply, preset]);
  const revert = useCallback(() => layout.reset(), [layout]);
  const save = useCallback(() => layout.savePreset(preset.id), [layout, preset.id]);
  // Unmount the menu in the same commit the dialog mounts; zag's own close lands a commit later and its layer
  // teardown would dismiss the dialog as nested above it.
  const edit = useCallback(() => {
    onClose();
    onEdit(preset);
  }, [onClose, onEdit, preset]);
  const remove = useCallback(() => {
    onClose();
    onDelete(preset);
  }, [onClose, onDelete, preset]);
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

  // Anchored to a measured rect rather than a trigger element: the chevron lives
  // inside the tab button and cannot be a `Menu.Trigger` of its own without
  // nesting buttons, and a right-click anchors at the pointer.
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
            <HStack justify="space-between" px="3" py="2">
              <MiddleTruncate fontSize="md" fontWeight="700" text={preset.label} />
              {showDrift ? (
                <Text color="fg.muted" fontSize="xs" flexShrink={0}>
                  {t('topbar.presets.unsaved')}
                </Text>
              ) : null}
            </HStack>
            <Menu.Separator />

            {isActive ? null : (
              <Menu.Item value="apply-preset" onClick={apply}>
                <Icon as={ArrowRightIcon} boxSize="3.5" />
                <Menu.ItemText>{t('topbar.presets.switch')}</Menu.ItemText>
              </Menu.Item>
            )}

            {isActive ? (
              <>
                {showDrift ? (
                  <Menu.Item value="revert-layout" onClick={revert}>
                    <Icon as={RotateCcwIcon} boxSize="3.5" />
                    <Menu.ItemText>{t('topbar.presets.revert')}</Menu.ItemText>
                  </Menu.Item>
                ) : null}
                <Menu.Item value="save-layout" onClick={save}>
                  <Icon as={SaveIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('topbar.presets.saveChanges')}</Menu.ItemText>
                  {saveShortcut ? (
                    <Text color="fg.subtle" fontSize="xs" ms="auto">
                      {saveShortcut}
                    </Text>
                  ) : null}
                </Menu.Item>
              </>
            ) : null}

            <Menu.Item value="edit-preset" onClick={edit}>
              <Icon as={PencilIcon} boxSize="3.5" />
              <Menu.ItemText>{t('topbar.presets.editWithEllipsis')}</Menu.ItemText>
            </Menu.Item>
            {isCustom ? (
              <Menu.Item data-danger="" value="delete-preset" onClick={remove}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                <Menu.ItemText>{t('topbar.presets.deleteWithEllipsis')}</Menu.ItemText>
              </Menu.Item>
            ) : null}

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
