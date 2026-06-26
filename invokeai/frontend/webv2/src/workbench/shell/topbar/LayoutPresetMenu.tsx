import type { LayoutPreset } from '@workbench/types';
import type { MouseEvent } from 'react';

import { HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, ConfirmDialog, IconButton, MenuContent, RenameDialog } from '@workbench/components/ui';
import { layoutPresets } from '@workbench/layoutPresets';
import { areLayoutPresetSnapshotsEqual, createLayoutPresetSnapshot } from '@workbench/layoutPresetSnapshots';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CheckIcon, ChevronDownIcon, EllipsisVerticalIcon, PencilIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';

interface PresetActionTarget {
  preset: LayoutPreset;
  x?: number;
  y?: number;
}

const getDefaultCustomPresetName = (presets: LayoutPreset[]): string => `Custom layout ${presets.length + 1}`;

const createCustomPresetId = (): string =>
  `custom-layout-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const TRIGGER_HOVER_PROPS = { bg: 'bg.muted' };
const DISABLED_PROPS = { opacity: 0.45 };

/** Global layout preset registry surfaced as a menu. */
export const LayoutPresetMenu = () => {
  const activeLayoutSnapshot = useActiveProjectSelector(createLayoutPresetSnapshot, areLayoutPresetSnapshotsEqual);
  const customPresets = useWorkbenchSelector((snapshot) => snapshot.state.account.customLayoutPresets ?? []);
  const dispatch = useWorkbenchDispatch();
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<LayoutPreset | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<LayoutPreset | null>(null);
  const [actionTarget, setActionTarget] = useState<PresetActionTarget | null>(null);
  const allPresets = useMemo(() => [...layoutPresets, ...customPresets], [customPresets]);
  const matchingPreset = allPresets.find((preset) =>
    areLayoutPresetSnapshotsEqual(activeLayoutSnapshot, preset.snapshot)
  );
  const canAddCurrentLayout = !matchingPreset;
  const triggerLabel = matchingPreset?.label ?? 'Custom';

  const addPreset = useCallback(
    (label: string) => {
      dispatch({ label, presetId: createCustomPresetId(), type: 'addLayoutPreset' });
    },
    [dispatch]
  );

  const renamePreset = useCallback(
    (preset: LayoutPreset, label: string) => {
      dispatch({ label, presetId: preset.id, type: 'renameLayoutPreset' });
    },
    [dispatch]
  );

  const deletePreset = useCallback(
    (preset: LayoutPreset) => {
      dispatch({ presetId: preset.id, type: 'deleteLayoutPreset' });
    },
    [dispatch]
  );
  const applyPreset = useCallback(
    (preset: LayoutPreset) => dispatch({ presetId: preset.id, type: 'applyPreset' }),
    [dispatch]
  );
  const openActionMenu = useCallback((preset: LayoutPreset, event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setActionTarget({ preset, x: event.clientX, y: event.clientY });
  }, []);
  const handleAddCurrentLayout = useCallback(() => {
    if (canAddCurrentLayout) {
      setIsAddOpen(true);
    }
  }, [canAddCurrentLayout]);
  const closeActionMenu = useCallback(() => setActionTarget(null), []);
  const openDeleteDialog = useCallback((preset: LayoutPreset) => setDeleteTarget(preset), []);
  const openRenameDialog = useCallback((preset: LayoutPreset) => setRenameTarget(preset), []);
  const closeAddDialog = useCallback(() => setIsAddOpen(false), []);
  const closeRenameDialog = useCallback(() => setRenameTarget(null), []);
  const submitRename = useCallback(
    (label: string) => {
      if (renameTarget) {
        renamePreset(renameTarget, label);
      }
    },
    [renamePreset, renameTarget]
  );
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  const confirmDelete = useCallback(() => {
    if (deleteTarget) {
      deletePreset(deleteTarget);
    }
  }, [deletePreset, deleteTarget]);

  return (
    <>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <Button
            bg="bg.subtle"
            borderWidth="1px"
            borderColor="border.emphasized"
            color="fg"
            fontSize="xs"
            fontWeight="500"
            justifyContent="space-between"
            size="xs"
            variant="outline"
            w="9rem"
            _hover={TRIGGER_HOVER_PROPS}
          >
            {triggerLabel}
            <Icon as={ChevronDownIcon} boxSize="3" />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent maxW="20rem" minW="18rem">
              <PresetGroup
                label="Built-in presets"
                matchingPreset={matchingPreset}
                presets={layoutPresets}
                onApply={applyPreset}
              />
              {customPresets.length ? (
                <>
                  <Menu.Separator />
                  <PresetGroup
                    isCustom
                    label="Custom presets"
                    matchingPreset={matchingPreset}
                    presets={customPresets}
                    onAction={openActionMenu}
                    onApply={applyPreset}
                    onContextMenu={openActionMenu}
                  />
                </>
              ) : null}
              <Menu.Separator />
              <Menu.Item
                value="add-current-layout"
                disabled={!canAddCurrentLayout}
                _disabled={DISABLED_PROPS}
                onClick={handleAddCurrentLayout}
              >
                <Icon as={PlusIcon} boxSize="3.5" />
                <Menu.ItemText>Add current layout...</Menu.ItemText>
                {!canAddCurrentLayout ? (
                  <Text color="fg.subtle" fontSize="2xs" ms="auto">
                    Saved
                  </Text>
                ) : null}
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <PresetActionMenu
        target={actionTarget}
        onClose={closeActionMenu}
        onDelete={openDeleteDialog}
        onRename={openRenameDialog}
      />
      <RenameDialog
        initialName={getDefaultCustomPresetName(customPresets)}
        isOpen={isAddOpen}
        label="Preset name"
        submitLabel="Add preset"
        submitUnchanged
        title="Add layout preset"
        onClose={closeAddDialog}
        onSubmit={addPreset}
      />
      {renameTarget ? (
        <RenameDialog
          initialName={renameTarget.label}
          isOpen={renameTarget !== null}
          label="Preset name"
          title="Rename layout preset"
          onClose={closeRenameDialog}
          onSubmit={submitRename}
        />
      ) : null}
      <ConfirmDialog
        body={`Delete "${deleteTarget?.label ?? 'this layout preset'}"? This only removes the saved preset.`}
        confirmLabel="Delete preset"
        isOpen={deleteTarget !== null}
        title="Delete layout preset?"
        onClose={closeDeleteDialog}
        onConfirm={confirmDelete}
      />
    </>
  );
};

const PresetGroup = ({
  isCustom = false,
  label,
  matchingPreset,
  onAction,
  onApply,
  onContextMenu,
  presets,
}: {
  isCustom?: boolean;
  label: string;
  matchingPreset?: LayoutPreset;
  onAction?: (preset: LayoutPreset, event: MouseEvent<HTMLElement>) => void;
  onApply: (preset: LayoutPreset) => void;
  onContextMenu?: (preset: LayoutPreset, event: MouseEvent<HTMLElement>) => void;
  presets: LayoutPreset[];
}) => (
  <Menu.ItemGroup>
    <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
      {label}
    </Menu.ItemGroupLabel>
    {presets.map((preset) => (
      <PresetMenuItem
        key={preset.id}
        isCustom={isCustom}
        isSelected={preset.id === matchingPreset?.id}
        preset={preset}
        onAction={onAction}
        onApply={onApply}
        onContextMenu={onContextMenu}
      />
    ))}
  </Menu.ItemGroup>
);

const PresetMenuItem = ({
  isCustom,
  isSelected,
  onAction,
  onApply,
  onContextMenu,
  preset,
}: {
  isCustom: boolean;
  isSelected: boolean;
  preset: LayoutPreset;
  onAction?: (preset: LayoutPreset, event: MouseEvent<HTMLElement>) => void;
  onApply: (preset: LayoutPreset) => void;
  onContextMenu?: (preset: LayoutPreset, event: MouseEvent<HTMLElement>) => void;
}) => {
  const handleApply = useCallback(() => onApply(preset), [onApply, preset]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => onContextMenu?.(preset, event),
    [onContextMenu, preset]
  );
  const handleAction = useCallback((event: MouseEvent<HTMLElement>) => onAction?.(preset, event), [onAction, preset]);

  return (
    <Menu.Item value={preset.id} onClick={handleApply} onContextMenu={handleContextMenu}>
      <Stack gap="0" flex="1" minW="0">
        <Text fontSize="xs" fontWeight="600">
          {preset.label}
        </Text>
      </Stack>
      <HStack flexShrink={0} gap="1">
        {isSelected ? <Icon as={CheckIcon} boxSize="3" color="accent.solid" /> : null}
        {isCustom ? (
          <IconButton
            aria-label={`Actions for ${preset.label}`}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={handleAction}
          >
            <EllipsisVerticalIcon />
          </IconButton>
        ) : null}
      </HStack>
    </Menu.Item>
  );
};

const PresetActionMenu = ({
  onClose,
  onDelete,
  onRename,
  target,
}: {
  target: PresetActionTarget | null;
  onClose: () => void;
  onDelete: (preset: LayoutPreset) => void;
  onRename: (preset: LayoutPreset) => void;
}) => {
  const positioning = useMemo(
    () =>
      target?.x !== undefined && target.y !== undefined
        ? {
            getAnchorRect: () => ({ height: 1, width: 1, x: target.x as number, y: target.y as number }),
            placement: 'bottom-start' as const,
          }
        : MENU_POSITIONING,
    [target]
  );
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root
      key={target?.preset.id ?? 'closed'}
      lazyMount
      open={target !== null}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Menu.Positioner>
          {target ? (
            <MenuContent minW="12rem">
              <PresetActionMenuItem action="rename" preset={target.preset} onDelete={onDelete} onRename={onRename} />
              <PresetActionMenuItem action="delete" preset={target.preset} onDelete={onDelete} onRename={onRename} />
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const PresetActionMenuItem = ({
  action,
  onDelete,
  onRename,
  preset,
}: {
  action: 'delete' | 'rename';
  preset: LayoutPreset;
  onDelete: (preset: LayoutPreset) => void;
  onRename: (preset: LayoutPreset) => void;
}) => {
  const handleClick = useCallback(() => {
    if (action === 'rename') {
      onRename(preset);
    } else {
      onDelete(preset);
    }
  }, [action, onDelete, onRename, preset]);

  if (action === 'rename') {
    return (
      <Menu.Item value="rename-preset" onClick={handleClick}>
        <Icon as={PencilIcon} boxSize="3.5" />
        <Menu.ItemText>Rename...</Menu.ItemText>
      </Menu.Item>
    );
  }

  return (
    <Menu.Item color="fg.error" value="delete-preset" onClick={handleClick}>
      <Icon as={Trash2Icon} boxSize="3.5" />
      <Menu.ItemText>Delete...</Menu.ItemText>
    </Menu.Item>
  );
};
