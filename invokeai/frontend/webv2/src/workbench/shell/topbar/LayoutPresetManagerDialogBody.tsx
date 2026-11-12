/* oxlint-disable react-perf/jsx-no-jsx-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { LayoutPreset } from '@workbench/layoutContracts';

import { Box, Icon, Portal } from '@chakra-ui/react';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { SortableListItem, SortableListStack } from '@platform/ui/list/SortableListStack';
import { Tooltip } from '@platform/ui/Tooltip';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { getLayoutPresetLabel } from '@workbench/layoutPresets';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { PencilIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import { closeLayoutPresetManager, openLayoutPresetDelete, openLayoutPresetEdit } from './layoutPresetManagerStore';

/**
 * The edit and delete dialogs stack over the manager, and focus outside a modal that traps it is never the user
 * leaving. Under StrictMode a dialog mounted open traps focus twice, so on close one trap returns focus into the
 * closing edit dialog after its layer has left the stack; the manager must not read that as a reason to close.
 */
const keepOpenOnFocusOutside = (event: { preventDefault: () => void }) => event.preventDefault();

/** The account-wide layout preset editor, rendered lazily from the top bar. */
export const LayoutPresetManagerDialogBody = ({
  isOpen,
  onExitComplete,
}: {
  isOpen: boolean;
  onExitComplete: () => void;
}) => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  // Labels as shown: a built-in preset not renamed by the account reads in the current language.
  const presets = useMemo(
    () => getOrderedLayoutPresets(account).map((preset) => ({ ...preset, label: getLayoutPresetLabel(preset, t) })),
    [account, t]
  );
  const presetIds = useMemo(() => presets.map(({ id }) => id), [presets]);
  const overriddenPresetIds = useMemo(
    () =>
      new Set([
        ...Object.keys(account.layoutPresetMetadataOverrides ?? {}),
        ...Object.keys(account.layoutPresetOverrides ?? {}),
        ...Object.keys(account.layoutPresetRouteOverrides ?? {}),
      ]),
    [account.layoutPresetMetadataOverrides, account.layoutPresetOverrides, account.layoutPresetRouteOverrides]
  );

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      closeLayoutPresetManager();
    }
  }, []);

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      unmountOnExit
      onExitComplete={onExitComplete}
      onFocusOutside={keepOpenOnFocusOutside}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content maxW="30rem">
            <Dialog.Header>
              <Dialog.Title>{t('topbar.presets.manage')}</Dialog.Title>
              <Dialog.CloseTrigger asChild>
                <CloseButton />
              </Dialog.CloseTrigger>
            </Dialog.Header>
            <Dialog.Body>
              {/* Row surfaces bleed into the dialog padding so their content lines up with the header. */}
              <SortableListStack
                dividers
                items={presetIds}
                label={t('topbar.presets.list')}
                mx="-2"
                onReorder={layout.reorderPresets}
              >
                {presets.map((preset) => (
                  <PresetRow
                    key={preset.id}
                    isOverridden={preset.isBuiltIn === true && overriddenPresetIds.has(preset.id)}
                    preset={preset}
                  />
                ))}
              </SortableListStack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button variant="ghost" onClick={closeLayoutPresetManager}>
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
  const edit = useCallback(() => openLayoutPresetEdit(preset.id), [preset.id]);
  const deletePreset = useCallback(() => openLayoutPresetDelete(preset.id), [preset.id]);
  const restoreDefault = useCallback(() => layout.restorePresetDefault(preset.id), [layout, preset.id]);

  return (
    <SortableListItem
      actions={
        <>
          <IconButton
            aria-label={t('topbar.presets.editNamed', { name: preset.label })}
            size="sm"
            tooltip={t('topbar.presets.edit')}
            variant="ghost"
            onClick={edit}
          >
            <Icon as={PencilIcon} boxSize="3.5" />
          </IconButton>
          {isOverridden ? (
            <Tooltip content={t('topbar.presets.restore')} showArrow>
              <IconButton
                aria-label={t('topbar.presets.restoreNamed', { name: preset.label })}
                size="sm"
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
              size="sm"
              tooltip={t('topbar.presets.delete')}
              variant="ghost"
              onClick={deletePreset}
            >
              <Icon as={Trash2Icon} boxSize="3.5" />
            </IconButton>
          )}
        </>
      }
      badges={
        isOverridden ? (
          <Box
            aria-label={t('topbar.presets.edited')}
            bg="accent.solid"
            boxSize="1.5"
            flexShrink={0}
            role="img"
            rounded="full"
          />
        ) : null
      }
      handleLabel={t('topbar.presets.reorderNamed', { name: preset.label })}
      id={preset.id}
      leading={<Icon as={resolveLayoutPresetIcon(preset.iconId)} boxSize="4" color="fg.muted" flexShrink={0} />}
      title={preset.label}
    />
  );
};
