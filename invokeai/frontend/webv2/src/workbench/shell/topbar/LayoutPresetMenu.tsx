import type { LayoutPreset } from '@workbench/types';
import type { MouseEvent } from 'react';

import { HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, ConfirmDialog, IconButton, MenuContent, RenameDialog } from '@workbench/components/ui';
import { layoutPresets } from '@workbench/layoutPresets';
import { doesProjectMatchLayoutPreset } from '@workbench/layoutPresetSnapshots';
import { useActiveProject, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CheckIcon, ChevronDownIcon, EllipsisVerticalIcon, PencilIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { useState } from 'react';

interface PresetActionTarget {
  preset: LayoutPreset;
  x?: number;
  y?: number;
}

const getDefaultCustomPresetName = (presets: LayoutPreset[]): string => `Custom layout ${presets.length + 1}`;

const createCustomPresetId = (): string =>
  `custom-layout-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Global layout preset registry surfaced as a menu. */
export const LayoutPresetMenu = () => {
  const activeProject = useActiveProject();
  const customPresets = useWorkbenchSelector((snapshot) => snapshot.state.account.customLayoutPresets ?? []);
  const dispatch = useWorkbenchDispatch();
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<LayoutPreset | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<LayoutPreset | null>(null);
  const [actionTarget, setActionTarget] = useState<PresetActionTarget | null>(null);
  const allPresets = [...layoutPresets, ...customPresets];
  const matchingPreset = allPresets.find((preset) => doesProjectMatchLayoutPreset(activeProject, preset));
  const canAddCurrentLayout = !matchingPreset;
  const triggerLabel = matchingPreset?.label ?? 'Custom';

  const addPreset = (label: string) => {
    dispatch({ label, presetId: createCustomPresetId(), type: 'addLayoutPreset' });
  };

  const renamePreset = (preset: LayoutPreset, label: string) => {
    dispatch({ label, presetId: preset.id, type: 'renameLayoutPreset' });
  };

  const deletePreset = (preset: LayoutPreset) => {
    dispatch({ presetId: preset.id, type: 'deleteLayoutPreset' });
  };

  return (
    <>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
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
            _hover={{ bg: 'bg.muted' }}
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
                onApply={(preset) => dispatch({ presetId: preset.id, type: 'applyPreset' })}
              />
              {customPresets.length ? (
                <>
                  <Menu.Separator />
                  <PresetGroup
                    isCustom
                    label="Custom presets"
                    matchingPreset={matchingPreset}
                    presets={customPresets}
                    onAction={(preset, event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      setActionTarget({ preset, x: event.clientX, y: event.clientY });
                    }}
                    onApply={(preset) => dispatch({ presetId: preset.id, type: 'applyPreset' })}
                    onContextMenu={(preset, event) => {
                      event.preventDefault();
                      setActionTarget({ preset, x: event.clientX, y: event.clientY });
                    }}
                  />
                </>
              ) : null}
              <Menu.Separator />
              <Menu.Item
                value="add-current-layout"
                disabled={!canAddCurrentLayout}
                _disabled={{ opacity: 0.45 }}
                onClick={() => {
                  if (canAddCurrentLayout) {
                    setIsAddOpen(true);
                  }
                }}
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
        onClose={() => setActionTarget(null)}
        onDelete={(preset) => setDeleteTarget(preset)}
        onRename={(preset) => setRenameTarget(preset)}
      />
      <RenameDialog
        initialName={getDefaultCustomPresetName(customPresets)}
        isOpen={isAddOpen}
        label="Preset name"
        submitLabel="Add preset"
        submitUnchanged
        title="Add layout preset"
        onClose={() => setIsAddOpen(false)}
        onSubmit={addPreset}
      />
      {renameTarget ? (
        <RenameDialog
          initialName={renameTarget.label}
          isOpen={renameTarget !== null}
          label="Preset name"
          title="Rename layout preset"
          onClose={() => setRenameTarget(null)}
          onSubmit={(label) => renamePreset(renameTarget, label)}
        />
      ) : null}
      <ConfirmDialog
        body={`Delete "${deleteTarget?.label ?? 'this layout preset'}"? This only removes the saved preset.`}
        confirmLabel="Delete preset"
        isOpen={deleteTarget !== null}
        title="Delete layout preset?"
        onClose={() => setDeleteTarget(null)}
        onConfirm={() => {
          if (deleteTarget) {
            deletePreset(deleteTarget);
          }
        }}
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
  onAction?: (preset: LayoutPreset, event: MouseEvent<HTMLButtonElement>) => void;
  onApply: (preset: LayoutPreset) => void;
  onContextMenu?: (preset: LayoutPreset, event: MouseEvent<HTMLDivElement>) => void;
  presets: LayoutPreset[];
}) => (
  <Menu.ItemGroup>
    <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
      {label}
    </Menu.ItemGroupLabel>
    {presets.map((preset) => (
      <Menu.Item
        key={preset.id}
        value={preset.id}
        onClick={() => onApply(preset)}
        onContextMenu={(event) => onContextMenu?.(preset, event)}
      >
        <Stack gap="0" flex="1" minW="0">
          <Text fontSize="xs" fontWeight="600">
            {preset.label}
          </Text>
        </Stack>
        <HStack flexShrink={0} gap="1">
          {preset.id === matchingPreset?.id ? <Icon as={CheckIcon} boxSize="3" color="accent.solid" /> : null}
          {isCustom ? (
            <IconButton
              aria-label={`Actions for ${preset.label}`}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={(event) => onAction?.(preset, event)}
            >
              <EllipsisVerticalIcon />
            </IconButton>
          ) : null}
        </HStack>
      </Menu.Item>
    ))}
  </Menu.ItemGroup>
);

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
}) => (
  <Menu.Root
    key={target?.preset.id ?? 'closed'}
    lazyMount
    open={target !== null}
    positioning={
      target?.x !== undefined && target.y !== undefined
        ? {
            getAnchorRect: () => ({ height: 1, width: 1, x: target.x as number, y: target.y as number }),
            placement: 'bottom-start',
          }
        : { placement: 'bottom-end' }
    }
    unmountOnExit
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <Portal>
      <Menu.Positioner>
        {target ? (
          <MenuContent minW="12rem">
            <Menu.Item value="rename-preset" onClick={() => onRename(target.preset)}>
              <Icon as={PencilIcon} boxSize="3.5" />
              <Menu.ItemText>Rename...</Menu.ItemText>
            </Menu.Item>
            <Menu.Item color="fg.error" value="delete-preset" onClick={() => onDelete(target.preset)}>
              <Icon as={Trash2Icon} boxSize="3.5" />
              <Menu.ItemText>Delete...</Menu.ItemText>
            </Menu.Item>
          </MenuContent>
        ) : null}
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
