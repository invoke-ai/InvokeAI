/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon } from '@chakra-ui/react';
import { reloadCustomNodes } from '@features/nodes/data/api';
import { refreshCustomNodePacks } from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui';
import { RefreshCwIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

export const ReloadNodesButton = () => {
  const notify = useNotify();
  const { t } = useTranslation();
  const [isReloading, setIsReloading] = useState(false);

  const handleReload = async () => {
    setIsReloading(true);

    try {
      await reloadCustomNodes();
      await refreshCustomNodePacks();
      notify.success(t('nodes.customNodesReloaded'));
    } catch (error) {
      notify.error(t('nodes.reloadFailed'), getApiErrorMessage(error, t('nodes.couldNotReloadCustomNodes')));
    } finally {
      setIsReloading(false);
    }
  };

  return (
    <Button loading={isReloading} size="2xs" variant="ghost" onClick={() => void handleReload()}>
      <Icon as={RefreshCwIcon} boxSize="3.5" />
      {isReloading ? t('nodes.reloading') : t('common.reload')}
    </Button>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
