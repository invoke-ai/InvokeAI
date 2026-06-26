/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button } from '@workbench/components/ui';
import { reloadCustomNodes } from '@workbench/customNodes/api';
import { refreshCustomNodePacks } from '@workbench/customNodes/nodesStore';
import { useNotify } from '@workbench/useNotify';
import { RefreshCwIcon } from 'lucide-react';
import { useState } from 'react';

export const ReloadNodesButton = () => {
  const notify = useNotify();
  const [isReloading, setIsReloading] = useState(false);

  const handleReload = async () => {
    setIsReloading(true);

    try {
      await reloadCustomNodes();
      await refreshCustomNodePacks();
      notify.success('Custom nodes reloaded');
    } catch (error) {
      notify.error('Reload failed', getApiErrorMessage(error, 'Could not reload custom nodes.'));
    } finally {
      setIsReloading(false);
    }
  };

  return (
    <Button loading={isReloading} size="2xs" variant="ghost" onClick={() => void handleReload()}>
      <Icon as={RefreshCwIcon} boxSize="3.5" />
      {isReloading ? 'Reloading' : 'Reload'}
    </Button>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
