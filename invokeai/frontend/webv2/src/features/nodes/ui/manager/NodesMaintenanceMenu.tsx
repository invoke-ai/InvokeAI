/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon, Menu, Portal } from '@chakra-ui/react';
import {
  getCustomNodesSnapshot,
  refreshCustomNodePacks,
  useCustomNodesSelector,
} from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { ClipboardCopyIcon, MoreHorizontalIcon, RefreshCcwIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * Library maintenance beside the reload button: explicit list refresh (which
 * surfaces failures the store records silently while packs are loaded) and
 * copying the custom-nodes path.
 */
export const NodesMaintenanceMenu = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const customNodesPath = useCustomNodesSelector((snapshot) => snapshot.customNodesPath);
  const { run } = useScopedAction();

  const handleRefresh = () =>
    run(
      async (owner) => {
        await refreshCustomNodePacks(owner);

        assertAccountScopeCurrent(owner);
        // The store records refresh failures without dropping loaded packs
        // (the list keeps rendering); an explicit user refresh still
        // deserves a failure toast.
        const { error } = getCustomNodesSnapshot();

        if (error !== null) {
          throw new Error(error);
        }
      },
      (message) => notify.error(t('nodes.refreshFailed'), message)
    );

  const handleCopyPath = async () => {
    if (customNodesPath === null) {
      return;
    }

    try {
      await navigator.clipboard.writeText(customNodesPath);
      notify.success(t('nodes.pathCopied'));
    } catch {
      notify.error(t('common.couldNotCopy'));
    }
  };

  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('nodes.libraryMaintenance')} size="2xs" variant="ghost">
          <Icon as={MoreHorizontalIcon} boxSize="4" />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem">
            <Menu.Item value="refresh" onClick={() => void handleRefresh()}>
              <Icon as={RefreshCcwIcon} boxSize="3.5" />
              <Menu.ItemText fontSize="xs">{t('nodes.refreshList')}</Menu.ItemText>
            </Menu.Item>
            <Menu.Item disabled={customNodesPath === null} value="copy-path" onClick={() => void handleCopyPath()}>
              <Icon as={ClipboardCopyIcon} boxSize="3.5" />
              <Menu.ItemText fontSize="xs">{t('nodes.copyPath')}</Menu.ItemText>
            </Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
