import { HStack } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';

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
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);
