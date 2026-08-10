/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { OrphanedModelInfo } from '@features/models/core/types';

import { Checkbox, Dialog, Flex, Icon, Menu, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { formatBytes } from '@features/models/core/taxonomy';
import { deleteOrphanedModels, emptyModelCache, getOrphanedModels } from '@features/models/data/api';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { useScopedAction } from '@features/models/ui/shared/useScopedAction';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { Button, CloseButton, ConfirmDialog, IconButton, MenuContent, Panel } from '@platform/ui';
import { BrushCleaningIcon, FolderSearchIcon, MoreHorizontalIcon, RefreshCcwIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Library maintenance: refresh, clean up orphaned model folders (files on disk
 * with no database record), and empty the in-memory model cache.
 */
export const MaintenanceMenu = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [isSyncDialogOpen, setIsSyncDialogOpen] = useState(false);
  const [isEmptyCacheConfirmOpen, setIsEmptyCacheConfirmOpen] = useState(false);
  // Separate instances: run ignores re-entry, and a slow cache emptying must
  // not swallow a refresh (or vice versa).
  const { run: runEmptyCache } = useScopedAction();
  const { run: runRefresh } = useScopedAction();

  const handleEmptyCache = () =>
    runEmptyCache(
      async (owner) => {
        await emptyModelCache(owner.signal);

        assertAccountScopeCurrent(owner);
        notify.success(t('models.cacheEmptied'));
      },
      (message) => notify.error(t('models.failedToEmptyCache'), message)
    );

  const handleRefresh = () =>
    runRefresh(
      async (owner) => {
        await refreshModels(owner);

        assertAccountScopeCurrent(owner);
        // refreshModels records failures in the snapshot instead of
        // rejecting (background refreshes stay silent by design); an
        // explicit user refresh still deserves a failure toast.
        const { error } = getModelsSnapshot();

        if (error !== null) {
          throw new Error(error);
        }
      },
      (message) => notify.error(t('models.refreshFailed'), message)
    );

  return (
    <>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('models.libraryMaintenance')} size="2xs" variant="ghost">
            <Icon as={MoreHorizontalIcon} boxSize="4" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="14rem">
              <Menu.Item value="refresh" onClick={() => void handleRefresh()}>
                <Icon as={RefreshCcwIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">{t('models.refreshList')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="sync" onClick={() => setIsSyncDialogOpen(true)}>
                <Icon as={FolderSearchIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">{t('models.cleanupOrphaned')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="empty-cache" onClick={() => setIsEmptyCacheConfirmOpen(true)}>
                <Icon as={BrushCleaningIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">{t('models.emptyCache')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {isSyncDialogOpen ? <OrphanedModelsDialog onClose={() => setIsSyncDialogOpen(false)} /> : null}
      <ConfirmDialog
        body={t('models.emptyCacheConfirmBody')}
        confirmLabel={t('models.emptyCache')}
        isDestructive={false}
        isOpen={isEmptyCacheConfirmOpen}
        title={t('models.emptyCacheConfirmTitle')}
        onClose={() => setIsEmptyCacheConfirmOpen(false)}
        onConfirm={async () => {
          await handleEmptyCache();
        }}
      />
    </>
  );
};

const OrphanedModelsDialog = ({ onClose }: { onClose: () => void }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [orphans, setOrphans] = useState<OrphanedModelInfo[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Nothing pre-selected: deleting files on disk must be an explicit choice.
  const [selectedPaths, setSelectedPaths] = useState<ReadonlySet<string>>(new Set());
  const { isBusy: isDeleting, run } = useScopedAction();

  const loadOrphans = async (owner: AccountScope): Promise<void> => {
    try {
      const result = await getOrphanedModels(owner.signal);

      if (isAccountScopeCurrent(owner)) {
        setOrphans(result);
      }
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        setLoadError(error instanceof Error ? error.message : t('models.failedToScanOrphaned'));
      }
    }
  };

  useMountEffect(() => {
    void loadOrphans(captureAccountScope());
  });

  const togglePath = (path: string) => {
    setSelectedPaths((current) => {
      const next = new Set(current);

      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }

      return next;
    });
  };

  const handleDelete = () =>
    run(
      async (owner) => {
        const result = await deleteOrphanedModels([...selectedPaths], owner.signal);

        assertAccountScopeCurrent(owner);
        const errorCount = Object.keys(result.errors).length;

        void refreshModels(owner);

        if (errorCount > 0) {
          notify.error(
            t('models.orphanedCleanup'),
            t('models.orphanedCleanupPartialDescription', {
              deleted: result.deleted.length,
              error: Object.values(result.errors)[0],
              failed: errorCount,
            })
          );
          // Keep the dialog open on the rescanned remainder so the user can
          // see which paths are left and retry.
          const deleted = new Set(result.deleted);

          setSelectedPaths((current) => new Set([...current].filter((path) => !deleted.has(path))));
          setOrphans(null);
          await loadOrphans(owner);

          return;
        }

        notify.success(
          t('models.orphanedCleanup'),
          t('models.orphanedDeletedDescription', { count: result.deleted.length })
        );
        onClose();
      },
      (message) => notify.error(t('models.orphanedCleanupFailed'), message)
    );

  const totalSelectedBytes =
    orphans?.filter((orphan) => selectedPaths.has(orphan.path)).reduce((sum, orphan) => sum + orphan.size_bytes, 0) ??
    0;

  return (
    <Dialog.Root
      open
      placement="center"
      scrollBehavior="inside"
      size="md"
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title fontSize="sm" fontWeight="700">
                  {t('models.orphanedTitle')}
                </Dialog.Title>
                <Text color="fg.subtle" fontSize="2xs">
                  {t('models.orphanedDescription')}
                </Text>
              </Stack>
            </Dialog.Header>
            <Dialog.Body>
              {loadError ? (
                <Text color="fg.error" fontSize="xs" py="4">
                  {loadError}
                </Text>
              ) : orphans === null ? (
                <Flex align="center" justify="center" py="8">
                  <Spinner color="fg.subtle" size="sm" />
                </Flex>
              ) : orphans.length === 0 ? (
                <Text color="fg.muted" fontSize="xs" py="4" textAlign="center">
                  {t('models.noOrphaned')}
                </Text>
              ) : (
                <Stack gap="1.5" py="2">
                  <Checkbox.Root
                    checked={
                      selectedPaths.size === 0 ? false : selectedPaths.size === orphans.length ? true : 'indeterminate'
                    }
                    colorPalette="accent"
                    ps="2"
                    size="sm"
                    onCheckedChange={() => {
                      setSelectedPaths(
                        selectedPaths.size === orphans.length
                          ? new Set()
                          : new Set(orphans.map((orphan) => orphan.path))
                      );
                    }}
                  >
                    <Checkbox.HiddenInput />
                    <Checkbox.Control />
                    <Checkbox.Label color="fg.muted" fontSize="2xs">
                      {t('models.selectAllCount', { count: orphans.length })}
                    </Checkbox.Label>
                  </Checkbox.Root>
                  {orphans.map((orphan) => (
                    <Panel key={orphan.path} alignItems="center" flexDirection="row" gap="2" p="2" tone="control">
                      <Checkbox.Root
                        checked={selectedPaths.has(orphan.path)}
                        colorPalette="accent"
                        size="sm"
                        onCheckedChange={() => togglePath(orphan.path)}
                      >
                        <Checkbox.HiddenInput />
                        <Checkbox.Control />
                      </Checkbox.Root>
                      <Stack flex="1" gap="0" minW="0">
                        <Text fontSize="2xs" fontWeight="600" overflowWrap="anywhere">
                          {orphan.path}
                        </Text>
                        <Text color="fg.subtle" fontSize="2xs">
                          {t('models.fileCount', { count: orphan.files.length })} · {formatBytes(orphan.size_bytes)}
                        </Text>
                      </Stack>
                    </Panel>
                  ))}
                </Stack>
              )}
            </Dialog.Body>
            <Dialog.Footer gap="2">
              <Button disabled={isDeleting} size="xs" variant="ghost" onClick={onClose}>
                {t('common.close')}
              </Button>
              <Button
                colorPalette="red"
                disabled={selectedPaths.size === 0 || !orphans || orphans.length === 0}
                loading={isDeleting}
                size="xs"
                variant="solid"
                onClick={() => void handleDelete()}
              >
                {t('models.deleteSelectedWithBytes', { bytes: formatBytes(totalSelectedBytes) })}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
