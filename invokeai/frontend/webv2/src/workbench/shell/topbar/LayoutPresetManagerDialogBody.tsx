import type { DragEndEvent, KeyboardSensorOptions } from '@dnd-kit/core';
import type { LayoutPreset, LayoutPresetId } from '@workbench/layoutContracts';

import { Box, Dialog, HStack, Icon, Portal, Stack, Text } from '@chakra-ui/react';
import { closestCenter, DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers';
import {
  sortableKeyboardCoordinates,
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { GripVerticalIcon, PencilIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { closeLayoutPresetManager, openLayoutPresetDelete, openLayoutPresetEdit } from './layoutPresetManagerStore';

const DND_MODIFIERS = [restrictToVerticalAxis, restrictToParentElement];
const POINTER_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;
const KEYBOARD_SENSOR_OPTIONS = {
  coordinateGetter: sortableKeyboardCoordinates,
} satisfies KeyboardSensorOptions;

/** The account-wide layout preset editor, rendered lazily from the top bar. */
export const LayoutPresetManagerDialogBody = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const presets = getOrderedLayoutPresets(account);
  const presetIds = presets.map(({ id }) => id);
  const overriddenPresetIds = new Set([
    ...Object.keys(account.layoutPresetMetadataOverrides ?? {}),
    ...Object.keys(account.layoutPresetOverrides ?? {}),
    ...Object.keys(account.layoutPresetRouteOverrides ?? {}),
  ]);
  const sensors = useSensors(
    useSensor(PointerSensor, POINTER_SENSOR_OPTIONS),
    useSensor(KeyboardSensor, KEYBOARD_SENSOR_OPTIONS)
  );

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      closeLayoutPresetManager();
    }
  }, []);
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
    <Dialog.Root open lazyMount unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content maxW="30rem">
            <Dialog.Header>
              <Dialog.Title>{t('topbar.presets.manage')}</Dialog.Title>
              <Dialog.CloseTrigger asChild>
                <CloseButton size="sm" />
              </Dialog.CloseTrigger>
            </Dialog.Header>
            <Dialog.Body>
              <DndContext
                collisionDetection={closestCenter}
                modifiers={DND_MODIFIERS}
                sensors={sensors}
                onDragEnd={handleDragEnd}
              >
                <SortableContext items={presetIds} strategy={verticalListSortingStrategy}>
                  <Stack gap="1.5">
                    {presets.map((preset) => (
                      <PresetRow
                        key={preset.id}
                        isOverridden={preset.isBuiltIn === true && overriddenPresetIds.has(preset.id)}
                        preset={preset}
                      />
                    ))}
                  </Stack>
                </SortableContext>
              </DndContext>
            </Dialog.Body>
            <Dialog.Footer>
              <Button size="sm" variant="outline" onClick={closeLayoutPresetManager}>
                {t('common.done')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const PresetRow = ({ isOverridden, preset }: { isOverridden: boolean; preset: LayoutPreset }) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const icon = resolveLayoutPresetIcon(preset.iconId);
  const { attributes, isDragging, listeners, setActivatorNodeRef, setNodeRef, transform, transition } = useSortable({
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
  const edit = useCallback(() => openLayoutPresetEdit(preset.id), [preset.id]);
  const deletePreset = useCallback(() => openLayoutPresetDelete(preset.id), [preset.id]);
  const restoreDefault = useCallback(() => layout.restorePresetDefault(preset.id), [layout, preset.id]);

  return (
    <HStack
      ref={setNodeRef}
      borderColor="border.subtle"
      borderWidth="1px"
      data-layout-preset-id={preset.id}
      gap="2"
      px="3"
      py="2"
      rounded="md"
      style={dndStyle}
    >
      <IconButton
        ref={setActivatorNodeRef}
        {...attributes}
        {...listeners}
        aria-label={t('topbar.presets.reorderNamed', { name: preset.label })}
        color="fg.subtle"
        cursor={isDragging ? 'grabbing' : 'grab'}
        size="2xs"
        touchAction="none"
        variant="ghost"
      >
        <Icon as={GripVerticalIcon} boxSize="3.5" />
      </IconButton>
      <Icon as={icon} boxSize="4" color="fg.muted" flexShrink={0} />
      <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
        {preset.label}
      </Text>
      {isOverridden ? (
        <Box
          aria-label={t('topbar.presets.edited')}
          bg="accent.solid"
          boxSize="1.5"
          flexShrink={0}
          role="img"
          rounded="full"
        />
      ) : null}
      <IconButton
        aria-label={t('topbar.presets.editNamed', { name: preset.label })}
        size="2xs"
        variant="ghost"
        onClick={edit}
      >
        <Icon as={PencilIcon} boxSize="3.5" />
      </IconButton>
      {isOverridden ? (
        <Tooltip content={t('topbar.presets.restore')} showArrow>
          <IconButton
            aria-label={t('topbar.presets.restoreNamed', { name: preset.label })}
            size="2xs"
            variant="ghost"
            onClick={restoreDefault}
          >
            <Icon as={RotateCcwIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      {preset.isBuiltIn ? null : (
        <IconButton
          aria-label={t('topbar.presets.deleteNamed', { name: preset.label })}
          color="fg.error"
          size="2xs"
          variant="ghost"
          onClick={deletePreset}
        >
          <Icon as={Trash2Icon} boxSize="3.5" />
        </IconButton>
      )}
    </HStack>
  );
};
