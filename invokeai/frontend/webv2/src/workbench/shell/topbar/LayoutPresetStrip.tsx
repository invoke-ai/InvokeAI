import type { LayoutPreset } from '@workbench/layoutContracts';
import type { KeyboardEvent, MouseEvent } from 'react';

import { Box, HStack, Icon, Menu, Portal, Text, VisuallyHidden } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tabs } from '@platform/ui/Tabs';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  createLayoutPresetActivator,
  loadLayoutPresetWidgets,
  preloadLayoutPresetWidgets,
} from '@workbench/layoutPresetActivation';
import { layoutPresets } from '@workbench/layoutPresets';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
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

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { openLayoutPresetDelete, openLayoutPresetEdit, openLayoutPresetManager } from './layoutPresetManagerStore';
import { getLayoutPresetPresentation } from './layoutPresetPresentation';
import { getPresetAccessibleName, getTopbarPresetTabs } from './layoutPresetStripModel';
import { HIDE_BELOW_PRESET_LABEL_WIDTH } from './topbarBreakpoints';
import { useLayoutDrift } from './useLayoutDrift';
import { useTopbarShortcut } from './useTopbarShortcut';

/** Marks the chevron inside a tab, so the tab's own click handler can tell them apart. */
const PRESET_MENU_ATTRIBUTE = 'data-preset-menu';
const PRESET_SCROLL_CSS = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' } as const;

const createCustomPresetId = (): string =>
  `custom-layout-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * The centre of the top bar: the layout presets as a tab strip.
 *
 * Tabs rather than a segmented control because the active preset carries its own
 * menu, and a tab is a container we can put that chevron inside — so the menu
 * belongs to the preset visually as well as conceptually, instead of floating
 * beside the group.
 *
 * A preset is an arrangement, never a widget, which is why the strip carries no
 * source or routing marker, and why a customised layout is not a separate entry
 * but the active entry with unsaved changes.
 */
export const LayoutPresetStrip = () => {
  const { t } = useTranslation();
  const { activePreset, hasDrifted } = useLayoutDrift();
  const { layout } = useWorkbenchCommands();
  const [isSaveAsOpen, setIsSaveAsOpen] = useState(false);
  const [menuTarget, setMenuTarget] = useState<{ anchor: DOMRect; preset: LayoutPreset } | null>(null);

  const customPresets = useWorkbenchSelector((snapshot) => snapshot.account.customLayoutPresets ?? []);
  const presets = useMemo(() => getTopbarPresetTabs(customPresets), [customPresets]);
  const activatePreset = useMemo(
    () => createLayoutPresetActivator({ apply: layout.applyPreset, load: loadLayoutPresetWidgets }),
    [layout.applyPreset]
  );

  const applyPreset = useCallback(
    (preset: LayoutPreset) => {
      if (preset.id !== activePreset.id) {
        void activatePreset(preset);
      }
    },
    [activatePreset, activePreset.id]
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
    ({ iconId, name }: { iconId: string; name: string }) => layout.createPreset(createCustomPresetId(), name, iconId),
    [layout]
  );
  const closeMenu = useCallback(() => setMenuTarget(null), []);
  const requestEdit = useCallback((preset: LayoutPreset) => openLayoutPresetEdit(preset.id), []);
  const requestDelete = useCallback((preset: LayoutPreset) => openLayoutPresetDelete(preset.id), []);

  return (
    <>
      <HStack gap="1" justify="center" maxW="min(36vw, 44rem)" minW="0">
        <Box css={PRESET_SCROLL_CSS} data-layout-preset-scroll="" maxW="full" minW="0" overflowX="auto">
          <Tabs.Root
            minW="max-content"
            size="sm"
            value={activePreset.id}
            variant="subtle"
            onValueChange={handleValueChange}
          >
            {/* The name belongs on the tablist, not the root — the root is a plain
                container and carries no role for it to name. */}
            <Tabs.List aria-label={t('topbar.presets.layoutPreset')} gap="0.5">
              {presets.map((preset) => (
                <PresetTab
                  key={preset.id}
                  hasDrifted={hasDrifted}
                  isActive={preset.id === activePreset.id}
                  preset={preset}
                  onOpenMenu={setMenuTarget}
                />
              ))}
            </Tabs.List>
            {/* The real "panel" is the dock itself, which lives outside this
                component and is shared by every preset. These stand in for it so
                each tab's `aria-controls` resolves to something that describes
                what selecting it did. */}
            {presets.map((preset) => (
              <Tabs.Content key={preset.id} value={preset.id} asChild>
                <VisuallyHidden>{getLayoutPresetPresentation(preset).tooltip}</VisuallyHidden>
              </Tabs.Content>
            ))}
          </Tabs.Root>
        </Box>

        {/* Saving the live arrangement as a new preset is the one preset action
            frequent enough to earn a place in the bar itself. */}
        <Tooltip content={t('topbar.presets.saveAsTooltip')} showArrow>
          <IconButton
            aria-label={t('topbar.presets.saveAsTooltip')}
            size="sm"
            variant="ghost"
            onClick={openSaveAsDialog}
          >
            <Icon as={PlusIcon} boxSize="4" />
          </IconButton>
        </Tooltip>
      </HStack>

      <PresetMenu
        isActive={menuTarget?.preset.id === activePreset.id}
        hasDrifted={hasDrifted}
        target={menuTarget}
        onApply={applyPreset}
        onClose={closeMenu}
        onDelete={requestDelete}
        onEdit={requestEdit}
      />

      <LayoutPresetDialog
        isOpen={isSaveAsOpen}
        name={`${activePreset.label} copy`}
        submitLabel={t('topbar.presets.save')}
        title={t('topbar.presets.saveAs')}
        onClose={closeSaveAsDialog}
        onSubmit={saveAsNewPreset}
      />
    </>
  );
};

const PresetTab = ({
  hasDrifted,
  isActive,
  onOpenMenu,
  preset,
}: {
  hasDrifted: boolean;
  isActive: boolean;
  preset: LayoutPreset;
  onOpenMenu: (target: { anchor: DOMRect; preset: LayoutPreset }) => void;
}) => {
  const { t } = useTranslation();
  const { icon, tooltip } = getLayoutPresetPresentation(preset);
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

  // Right-click opens the same menu, on whichever preset was under the pointer —
  // including inactive ones, so a custom preset can be renamed or deleted
  // without first switching into it.
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

  // Keyboard parity for the chevron. Left/Right already move between tabs, so
  // Down is free and is the usual "open this control's menu" key.
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (!isActive || event.key !== 'ArrowDown') {
        return;
      }

      event.preventDefault();
      onOpenMenu({ anchor: event.currentTarget.getBoundingClientRect(), preset });
    },
    [isActive, onOpenMenu, preset]
  );

  return (
    <Tooltip content={`${preset.label} — ${tooltip}`} showArrow>
      <Tabs.Trigger
        aria-label={getPresetAccessibleName(preset, showDrift, t('topbar.presets.unsaved'))}
        aria-keyshortcuts={isActive ? 'ArrowDown' : undefined}
        gap="1.5"
        value={preset.id}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onFocus={handlePreload}
        onKeyDown={handleKeyDown}
        onPointerEnter={handlePreload}
      >
        <Icon as={icon} boxSize="3.5" flexShrink={0} />
        {/* The explicit aria-label above keeps the tab named when this visible
            text is removed from the accessibility tree below 1280px. */}
        <Text as="span" css={HIDE_BELOW_PRESET_LABEL_WIDTH}>
          {preset.label}
        </Text>
        {/* The dot is the whole signal that a layout has diverged, so it has to
            reach screen readers too — as part of the tab's name. */}
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
    </Tooltip>
  );
};

const MENU_AFFORDANCE_HOVER_PROPS = { bg: 'bg.emphasized', color: 'fg' } as const;

/** Filled accent dot: this preset is loaded, but the live layout has moved on. */
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
  const isCustom = preset ? !layoutPresets.some((builtIn) => builtIn.id === preset.id) : false;
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
                <Text fontSize="xs" fontWeight="700" truncate>
                  {preset.label}
                </Text>
                {showDrift ? (
                  <Text color="fg.muted" fontSize="2xs" flexShrink={0}>
                    {t('topbar.presets.unsaved')}
                  </Text>
                ) : null}
              </HStack>
              <Menu.Separator />

              {/* Right-clicking a preset you are not in should still offer the
                  thing left-clicking it does, the way any list does. */}
              {isActive ? null : (
                <Menu.Item value="apply-preset" onClick={apply}>
                  <Icon as={ArrowRightIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('topbar.presets.switch')}</Menu.ItemText>
                </Menu.Item>
              )}

              {/* Revert and Save act on the *live* layout, so they only make
                  sense for the preset you are currently in. */}
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

              {/* Renaming, re-iconing, and deleting belong to the preset itself,
                  so they work on any custom preset without switching into it. */}
              {isCustom ? (
                <>
                  <Menu.Item value="edit-preset" onClick={edit}>
                    <Icon as={PencilIcon} boxSize="3.5" />
                    <Menu.ItemText>{t('topbar.presets.editWithEllipsis')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item color="fg.error" value="delete-preset" _hover={DELETE_HOVER_PROPS} onClick={remove}>
                    <Icon as={Trash2Icon} boxSize="3.5" />
                    <Menu.ItemText>{t('topbar.presets.deleteWithEllipsis')}</Menu.ItemText>
                  </Menu.Item>
                </>
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
