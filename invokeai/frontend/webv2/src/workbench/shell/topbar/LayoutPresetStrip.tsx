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
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayoutPresetDialogValue } from './layoutPresetDialogModel';

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { openLayoutPresetDelete, openLayoutPresetEdit, openLayoutPresetManager } from './layoutPresetManagerStore';
import { HIDE_BELOW_PRESET_LABEL_WIDTH } from './topbarBreakpoints';
import { useActiveLayoutPresetId, useLayoutDrift } from './useLayoutDrift';
import { useTopbarShortcut } from './useTopbarShortcut';

const PRESET_MENU_ATTRIBUTE = 'data-preset-menu';
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
  const [menuTarget, setMenuTarget] = useState<{ anchor: DOMRect; preset: LayoutPreset } | null>(null);

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
  /**
   * Desktop semantics: the control acknowledges the press on its own frame,
   * then the workspace rearranges. Activating inside the handler put a
   * ~100ms blocking React commit between the click and any pixel changing,
   * and the tab itself did not repaint for ~330ms.
   */
  const requestPreset = useCallback(
    (presetId: LayoutPresetId) => {
      setPendingPresetId(presetId);
      requestAnimationFrame(() => {
        void layout.activatePreset(presetId).then((appliedPresetId) => {
          // An activation can be dropped — superseded, overtaken by a project
          // switch, or aimed at a preset the account has since replaced. Give
          // the tab back to the store rather than leave it painting a preset
          // nothing ever applied. Guarded so a newer request keeps its own.
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
              size="xs"
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
                      onOpenMenu={setMenuTarget}
                      onRequest={requestPreset}
                    />
                  ))}
                </Tabs.List>
              </SortableContext>
              {/* The real "panel" is the dock itself, which lives outside this
                  component and is shared by every preset. These stand in for it so
                  each tab's `aria-controls` resolves to something that describes
                  what selecting it did. */}
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
            size="xs"
            variant="ghost"
            onClick={openSaveAsDialog}
          >
            <Icon as={PlusIcon} boxSize="4" />
          </IconButton>
        </Tooltip>
      </HStack>

      <PresetMenu
        isActive={menuTarget?.preset.id === selectedPresetId}
        hasDrifted={hasDrifted}
        target={menuTarget}
        onApply={applyPreset}
        onClose={closeMenu}
        onDelete={requestDelete}
        onEdit={requestEdit}
      />

      {isSaveAsOpen ? (
        <LayoutPresetDialog
          defaultRoute={saveAsDefaultRoute}
          isOpen
          name={`${activePreset.label} copy`}
          sourceOptions={sourceOptions}
          submitLabel={t('topbar.presets.save')}
          title={t('topbar.presets.saveAs')}
          onClose={closeSaveAsDialog}
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
  onOpenMenu: (target: { anchor: DOMRect; preset: LayoutPreset }) => void;
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
      transition,
      zIndex: isDragging ? 1 : undefined,
    }),
    [isDragging, transform, transition]
  );
  const handlePreload = useCallback(() => preloadLayoutPresetWidgets(preset), [preset]);
  const showDrift = isActive && hasDrifted;

  // A tab is a `<button>`, so the chevron cannot be one — nesting buttons is
  // invalid and the browser hoists the inner one out of the tab entirely. It is
  // a span the tab's own handler recognises, the same way the project tabs used
  // to carry their close affordance.
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

  // The press, not the click, is what the user is waiting on: the tabs machine
  // lands the selection correctly either way, but only once the button is
  // released. Primary button only — `pointerdown` fires for the secondary
  // button too, ahead of `contextmenu`, and right-clicking an inactive preset
  // to reach its menu must not switch to it first and strip the menu of the
  // very item it was opened for. dnd-kit's own `MouseSensor` bails the same way.
  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      // dnd-kit owns the drag gesture; chain rather than replace. Inert today —
      // the strip's sensors activate on `onMouseDown`/`onTouchStart`/`onKeyDown`
      // and only the unused `PointerSensor` would put a listener here — kept so
      // a future sensor swap does not silently lose its activator.
      (listeners as { onPointerDown?: (value: unknown) => void } | undefined)?.onPointerDown?.(event);

      if (event.button === 0 && !isActive) {
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
      cursor={isDragging ? 'grabbing' : 'pointer'}
      data-layout-preset-id={preset.id}
      gap="1.5"
      style={dndStyle}
      touchAction="pan-x"
      value={preset.id}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      onFocus={handlePreload}
      onKeyDown={handleKeyDown}
      onPointerDown={handlePointerDown}
      onPointerEnter={handlePreload}
    >
      <Icon as={icon} boxSize="3.5" flexShrink={0} />
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

const MENU_AFFORDANCE_HOVER_PROPS = { bg: 'bg.emphasized', color: 'fg' } as const;

const DriftDot = () => <Box aria-hidden="true" bg="accent.solid" boxSize="1.5" flexShrink={0} rounded="full" />;

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
  target: { anchor: DOMRect; preset: LayoutPreset } | null;
  onApply: (preset: LayoutPreset) => void;
  onClose: () => void;
  onDelete: (preset: LayoutPreset) => void;
  onEdit: (preset: LayoutPreset) => void;
}) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const saveShortcut = useTopbarShortcut('app.saveLayoutPreset');
  const preset = target?.preset;
  const isCustom = preset ? preset.isBuiltIn !== true : false;
  const showDrift = isActive && hasDrifted;

  const apply = useCallback(() => {
    if (preset) {
      onApply(preset);
    }
  }, [onApply, preset]);
  const revert = useCallback(() => layout.reset(), [layout]);
  const save = useCallback(() => {
    if (preset) {
      layout.savePreset(preset.id);
    }
  }, [layout, preset]);
  const edit = useCallback(() => {
    if (preset) {
      onEdit(preset);
    }
  }, [onEdit, preset]);
  const remove = useCallback(() => {
    if (preset) {
      onDelete(preset);
    }
  }, [onDelete, preset]);
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
  const anchor = target?.anchor ?? null;
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => (anchor ? { height: anchor.height, width: anchor.width, x: anchor.x, y: anchor.y } : null),
      placement: 'bottom-end' as const,
    }),
    [anchor]
  );

  return (
    <Menu.Root lazyMount open={target !== null} positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          {preset ? (
            <MenuContent minW="16rem">
              <HStack justify="space-between" px="3" py="2">
                <MiddleTruncate fontSize="xs" fontWeight="700" text={preset.label} />
                {showDrift ? (
                  <Text color="fg.muted" fontSize="2xs" flexShrink={0}>
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
                      <Text color="fg.subtle" fontSize="2xs" ms="auto">
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
                <Menu.Item color="fg.error" value="delete-preset" _hover={DELETE_HOVER_PROPS} onClick={remove}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText>{t('topbar.presets.deleteWithEllipsis')}</Menu.ItemText>
                </Menu.Item>
              ) : null}

              <Menu.Separator />
              <Menu.Item value="manage-presets" onClick={openLayoutPresetManager}>
                <Icon as={SettingsIcon} boxSize="3.5" />
                <Menu.ItemText>{t('topbar.presets.manage')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const DELETE_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' } as const;
