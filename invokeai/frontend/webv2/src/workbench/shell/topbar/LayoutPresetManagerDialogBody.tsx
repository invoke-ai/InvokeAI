import type { LayoutPreset } from '@workbench/layoutContracts';

import { Dialog, HStack, Icon, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { layoutPresets } from '@workbench/layoutPresets';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { PencilIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { closeLayoutPresetManager, openLayoutPresetDelete, openLayoutPresetEdit } from './layoutPresetManagerStore';
import { getLayoutPresetPresentation } from './layoutPresetPresentation';

/**
 * The long tail of preset administration, moved off the top bar.
 *
 * The strip itself only ever offers the three switch targets and the active
 * preset's own menu; renaming, deleting, and undoing a saved edit to a built-in
 * are rare enough that putting them in the bar would cost more than they return
 * — which is also why this body is only fetched once someone opens it.
 */
export const LayoutPresetManagerDialogBody = () => {
  const { t } = useTranslation();
  const customPresets = useWorkbenchSelector((snapshot) => snapshot.account.customLayoutPresets ?? []);
  const overriddenPresetIds = useWorkbenchSelector((snapshot) =>
    Object.keys(snapshot.account.layoutPresetOverrides ?? {})
  );

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      closeLayoutPresetManager();
    }
  }, []);

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
              <Stack gap="4">
                <Stack gap="1.5">
                  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                    {t('topbar.presets.builtIn')}
                  </Text>
                  {layoutPresets.map((preset) => (
                    <BuiltInPresetRow
                      key={preset.id}
                      isOverridden={overriddenPresetIds.includes(preset.id)}
                      preset={preset}
                    />
                  ))}
                </Stack>
                <Stack gap="1.5">
                  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                    {t('topbar.presets.custom')}
                  </Text>
                  {customPresets.length === 0 ? (
                    <Text color="fg.muted" fontSize="xs">
                      {t('topbar.presets.none')}
                    </Text>
                  ) : (
                    customPresets.map((preset) => (
                      <CustomPresetRow
                        key={preset.id}
                        preset={preset}
                        onDelete={openLayoutPresetDelete}
                        onEdit={openLayoutPresetEdit}
                      />
                    ))
                  )}
                </Stack>
              </Stack>
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

const BuiltInPresetRow = ({ isOverridden, preset }: { isOverridden: boolean; preset: LayoutPreset }) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const { icon, tooltip } = getLayoutPresetPresentation(preset);
  // Clearing the override restores the shipped arrangement. Saving the shipped
  // snapshot back is exactly that, and needs no second reducer path.
  const restoreDefault = useCallback(() => layout.restorePresetDefault(preset.id), [layout, preset.id]);

  return (
    <HStack borderColor="border.subtle" borderWidth="1px" gap="2" px="3" py="2" rounded="md">
      <Icon as={icon} boxSize="4" color="fg.muted" flexShrink={0} />
      <Stack flex="1" gap="0" minW="0">
        <Text fontSize="xs" fontWeight="600">
          {preset.label}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {isOverridden ? `${tooltip} · edited` : tooltip}
        </Text>
      </Stack>
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
    </HStack>
  );
};

const CustomPresetRow = ({
  onDelete,
  onEdit,
  preset,
}: {
  preset: LayoutPreset;
  onDelete: (presetId: string) => void;
  onEdit: (presetId: string) => void;
}) => {
  const { t } = useTranslation();
  const { icon } = getLayoutPresetPresentation(preset);
  const handleEdit = useCallback(() => onEdit(preset.id), [onEdit, preset.id]);
  const handleDelete = useCallback(() => onDelete(preset.id), [onDelete, preset.id]);

  return (
    <HStack borderColor="border.subtle" borderWidth="1px" gap="2" px="3" py="2" rounded="md">
      <Icon as={icon} boxSize="4" color="fg.muted" flexShrink={0} />
      <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
        {preset.label}
      </Text>
      <IconButton
        aria-label={t('topbar.presets.editNamed', { name: preset.label })}
        size="2xs"
        variant="ghost"
        onClick={handleEdit}
      >
        <Icon as={PencilIcon} boxSize="3.5" />
      </IconButton>
      <IconButton
        aria-label={t('topbar.presets.deleteNamed', { name: preset.label })}
        color="fg.error"
        size="2xs"
        variant="ghost"
        onClick={handleDelete}
      >
        <Icon as={Trash2Icon} boxSize="3.5" />
      </IconButton>
    </HStack>
  );
};
