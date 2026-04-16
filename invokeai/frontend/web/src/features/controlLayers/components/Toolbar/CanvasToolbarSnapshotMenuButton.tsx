import {
  ConfirmationAlertDialog,
  Flex,
  IconButton,
  Input,
  Menu,
  MenuButton,
  MenuDivider,
  MenuGroup,
  MenuItem,
  MenuList,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import type { SnapshotInfo } from 'features/controlLayers/hooks/useCanvasSnapshots';
import { useCanvasSnapshots } from 'features/controlLayers/hooks/useCanvasSnapshots';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { toast } from 'features/toast/toast';
import type { ChangeEvent, KeyboardEvent, MouseEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCameraBold, PiFloppyDiskBold, PiTrashBold } from 'react-icons/pi';

const SnapshotItem = memo(
  ({
    snapshot,
    onRestore,
    onDelete,
    isRestoreDisabled,
  }: {
    snapshot: SnapshotInfo;
    onRestore: (key: string, name: string) => void;
    onDelete: (e: MouseEvent, key: string, name: string) => void;
    isRestoreDisabled: boolean;
  }) => {
    const handleClick = useCallback(() => {
      onRestore(snapshot.key, snapshot.name);
    }, [onRestore, snapshot.key, snapshot.name]);

    const handleDelete = useCallback(
      (e: MouseEvent) => {
        onDelete(e, snapshot.key, snapshot.name);
      },
      [onDelete, snapshot.key, snapshot.name]
    );

    return (
      <MenuItem onClick={handleClick} isDisabled={isRestoreDisabled}>
        <Flex w="full" justifyContent="space-between" alignItems="center">
          <Text fontSize="sm" noOfLines={1}>
            {snapshot.name}
          </Text>
          <IconButton
            aria-label="Delete"
            icon={<PiTrashBold />}
            size="xs"
            variant="ghost"
            colorScheme="error"
            onClick={handleDelete}
            isDisabled={isRestoreDisabled}
          />
        </Flex>
      </MenuItem>
    );
  }
);

SnapshotItem.displayName = 'SnapshotItem';

const getDefaultSnapshotName = (): string => {
  const now = new Date();
  const y = now.getFullYear();
  const mo = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  const h = String(now.getHours()).padStart(2, '0');
  const mi = String(now.getMinutes()).padStart(2, '0');
  return `${y}-${mo}-${d} ${h}-${mi}`;
};

export const CanvasToolbarSnapshotMenuButton = memo(() => {
  const { t } = useTranslation();
  const { snapshots, saveSnapshot, restoreSnapshot, deleteSnapshot } = useCanvasSnapshots();
  const isStaging = useCanvasIsStaging();
  const [snapshotName, setSnapshotName] = useState('');
  const overwriteDialog = useDisclosure();
  const [pendingOverwriteName, setPendingOverwriteName] = useState<string | null>(null);

  const onNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSnapshotName(e.target.value);
  }, []);

  const doSave = useCallback(
    async (name: string) => {
      const success = await saveSnapshot(name);
      if (success) {
        toast({ title: t('controlLayers.snapshot.snapshotSaved', { name }), status: 'info' });
        setSnapshotName('');
      } else {
        toast({ title: t('controlLayers.snapshot.snapshotSaveFailed'), status: 'error' });
      }
    },
    [saveSnapshot, t]
  );

  const onSave = useCallback(async () => {
    const name = snapshotName.trim() || getDefaultSnapshotName();
    if (snapshots.some((s) => s.name === name)) {
      setPendingOverwriteName(name);
      overwriteDialog.onOpen();
      return;
    }
    await doSave(name);
  }, [snapshotName, snapshots, doSave, overwriteDialog]);

  const onConfirmOverwrite = useCallback(() => {
    if (pendingOverwriteName) {
      doSave(pendingOverwriteName);
      setPendingOverwriteName(null);
    }
  }, [pendingOverwriteName, doSave]);

  const onCloseOverwriteDialog = useCallback(() => {
    setPendingOverwriteName(null);
    overwriteDialog.onClose();
  }, [overwriteDialog]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        onSave();
      }
    },
    [onSave]
  );

  const onRestore = useCallback(
    async (key: string, name: string) => {
      const result = await restoreSnapshot(key);
      if (result.success) {
        if (result.missingImageCount && result.missingImageCount > 0) {
          toast({
            title: t('controlLayers.snapshot.snapshotRestored', { name }),
            description: t('controlLayers.snapshot.snapshotMissingImages', { count: result.missingImageCount }),
            status: 'warning',
          });
        } else {
          toast({ title: t('controlLayers.snapshot.snapshotRestored', { name }), status: 'info' });
        }
      } else if (result.error === 'incompatible') {
        toast({
          title: t('controlLayers.snapshot.snapshotRestoreFailed'),
          description: t('controlLayers.snapshot.snapshotIncompatible'),
          status: 'error',
        });
      } else {
        toast({ title: t('controlLayers.snapshot.snapshotRestoreFailed'), status: 'error' });
      }
    },
    [restoreSnapshot, t]
  );

  const onDelete = useCallback(
    async (e: MouseEvent, key: string, name: string) => {
      e.stopPropagation();
      const success = await deleteSnapshot(key);
      if (success) {
        toast({ title: t('controlLayers.snapshot.snapshotDeleted', { name }), status: 'info' });
      } else {
        toast({ title: t('controlLayers.snapshot.snapshotDeleteFailed'), status: 'error' });
      }
    },
    [deleteSnapshot, t]
  );

  return (
    <>
      <Menu placement="bottom-end" closeOnSelect={false}>
        <MenuButton
          as={IconButton}
          aria-label={t('controlLayers.snapshot.snapshots')}
          tooltip={t('controlLayers.snapshot.snapshots')}
          icon={<PiCameraBold />}
          variant="link"
          alignSelf="stretch"
        />
        <MenuList maxH="60vh" overflowY="auto">
          <MenuGroup title={t('controlLayers.snapshot.saveSnapshot')}>
            <Flex px={3} pb={2} gap={2} alignItems="center">
              <Input
                size="sm"
                placeholder={t('controlLayers.snapshot.snapshotNamePlaceholder')}
                value={snapshotName}
                onChange={onNameChange}
                onKeyDown={onKeyDown}
              />
              <IconButton
                aria-label={t('controlLayers.snapshot.save')}
                icon={<PiFloppyDiskBold />}
                size="sm"
                onClick={onSave}
              />
            </Flex>
          </MenuGroup>
          {snapshots.length > 0 && (
            <>
              <MenuDivider />
              <MenuGroup title={t('controlLayers.snapshot.restoreSnapshot')}>
                {snapshots.map((snapshot) => (
                  <SnapshotItem
                    key={snapshot.key}
                    snapshot={snapshot}
                    onRestore={onRestore}
                    onDelete={onDelete}
                    isRestoreDisabled={isStaging}
                  />
                ))}
              </MenuGroup>
            </>
          )}
        </MenuList>
      </Menu>
      <ConfirmationAlertDialog
        isOpen={overwriteDialog.isOpen}
        onClose={onCloseOverwriteDialog}
        title={t('controlLayers.snapshot.overwriteSnapshotTitle')}
        acceptCallback={onConfirmOverwrite}
        acceptButtonText={t('controlLayers.snapshot.overwrite')}
        useInert={false}
      >
        <Text>{t('controlLayers.snapshot.overwriteSnapshotMessage', { name: pendingOverwriteName ?? '' })}</Text>
      </ConfirmationAlertDialog>
    </>
  );
});

CanvasToolbarSnapshotMenuButton.displayName = 'CanvasToolbarSnapshotMenuButton';
