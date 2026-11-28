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
import { getActiveWorkspaceTab } from '@workbench/workspace/workspaceNavigation';
import { RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { resolveLayoutPresetIcon } from './layoutPresetIcons';
import {
  closeLayoutPresetManager,
  layoutPresetManagerStore,
  openLayoutPresetDelete,
  openLayoutPresetRestore,
} from './layoutPresetManagerStore';

/** The account's workspaces: their order, a built-in's default, deleting a custom one. Rendered lazily. */
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
      finalFocusEl={getReturnFocus}
      lazyMount
      open={isOpen}
      unmountOnExit
      onExitComplete={onExitComplete}
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

/** Back to the tab whose menu opened the manager, or the active one when that tab is gone (deleted from here). */
const getReturnFocus = (): HTMLElement | null => {
  const { openerPresetId } = layoutPresetManagerStore.getSnapshot();

  return (
    (openerPresetId === null
      ? null
      : document.querySelector<HTMLElement>(`[role="tab"][data-layout-preset-id="${CSS.escape(openerPresetId)}"]`)) ??
    getActiveWorkspaceTab()
  );
};

const PresetRow = ({ isOverridden, preset }: { isOverridden: boolean; preset: LayoutPreset }) => {
  const { t } = useTranslation();
  const deletePreset = useCallback(() => openLayoutPresetDelete(preset.id), [preset.id]);
  const restoreDefault = useCallback(() => openLayoutPresetRestore(preset.id), [preset.id]);

  return (
    <SortableListItem
      actions={
        <>
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
