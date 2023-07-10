import { HStack } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import LoadNodesButton from '../ui/LoadNodesButton';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import SaveNodesButton from '../ui/SaveNodesButton';

const TopCenterPanel = () => {
  const dispatch = useAppDispatch();

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <IAIButton onClick={handleReloadSchema}>Reload Schema</IAIButton>
        <SaveNodesButton />
        <LoadNodesButton />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);
