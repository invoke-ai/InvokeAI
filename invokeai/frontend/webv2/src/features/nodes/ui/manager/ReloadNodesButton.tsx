/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Icon } from '@chakra-ui/react';
import { reloadCustomNodes } from '@features/nodes/data/api';
import { getCustomNodesSnapshot, refreshCustomNodePacks } from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui';
import { RefreshCwIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ReloadNodesButton = () => {
  const notify = useNotify();
  const { t } = useTranslation();
  const { isBusy: isReloading, run } = useScopedAction();

  const handleReload = () =>
    run(
      async (owner) => {
        const { status } = await reloadCustomNodes(owner.signal);

        assertAccountScopeCurrent(owner);
        await refreshCustomNodePacks(owner);
        assertAccountScopeCurrent(owner);

        // The store records refetch failures without dropping loaded packs,
        // so the stale list keeps rendering; the reload's own toast must not
        // celebrate over it. (Reload is the only refresh affordance — a
        // separate "refresh list" action would just duplicate this one.)
        const { error: refreshError } = getCustomNodesSnapshot();

        if (refreshError !== null) {
          notify.error(t('nodes.refreshFailed'), refreshError);

          return;
        }

        // The backend reports the outcome as prose (custom_nodes.py). Only
        // its success phrasing earns a green toast; anything else — like
        // "No custom nodes directory found." — surfaces as-is. Unknown
        // statuses fail honest, never fail green.
        if (status.startsWith('Custom nodes reloaded')) {
          notify.success(t('nodes.customNodesReloaded'));
        } else {
          notify.warning(t('nodes.reloadNotice'), status);
        }
      },
      (_message, error) =>
        notify.error(t('nodes.reloadFailed'), getApiErrorMessage(error, t('nodes.couldNotReloadCustomNodes')))
    );

  return (
    <Button loading={isReloading} size="2xs" variant="ghost" onClick={() => void handleReload()}>
      <Icon as={RefreshCwIcon} boxSize="3.5" />
      {isReloading ? t('nodes.reloading') : t('common.reload')}
    </Button>
  );
};
