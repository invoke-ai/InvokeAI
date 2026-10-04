/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon, Menu, Portal } from '@chakra-ui/react';
import { emptyModelCache } from '@features/models/data/api';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { OrphanedModelsDialog } from '@features/models/ui/library/OrphanedModelsDialog';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ConfirmDialog, IconButton, MenuContent } from '@platform/ui';
import { BrushCleaningIcon, FolderSearchIcon, MoreHorizontalIcon, RefreshCcwIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

export const MaintenanceMenu = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [isSyncDialogOpen, setIsSyncDialogOpen] = useState(false);
  const syncDialog = useExitPresence(isSyncDialogOpen);
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
          <IconButton aria-label={t('models.libraryMaintenance')} size="sm" variant="ghost">
            <Icon as={MoreHorizontalIcon} boxSize="4" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="14rem">
              <Menu.Item value="refresh" onClick={() => void handleRefresh()}>
                <Icon as={RefreshCcwIcon} boxSize="3.5" />
                <Menu.ItemText>{t('models.refreshList')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="sync" onClick={() => setIsSyncDialogOpen(true)}>
                <Icon as={FolderSearchIcon} boxSize="3.5" />
                <Menu.ItemText>{t('models.cleanupOrphaned')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="empty-cache" onClick={() => setIsEmptyCacheConfirmOpen(true)}>
                <Icon as={BrushCleaningIcon} boxSize="3.5" />
                <Menu.ItemText>{t('models.emptyCache')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {syncDialog.isMounted ? (
        <OrphanedModelsDialog
          key={syncDialog.generation}
          isOpen={syncDialog.isOpen}
          onClose={() => setIsSyncDialogOpen(false)}
          onExitComplete={syncDialog.release}
        />
      ) : null}
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
