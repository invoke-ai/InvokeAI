import { HStack } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import NodeInvokeButton from '../ui/NodeInvokeButton';

const TopCenterPanel = () => {
  const dispatch = useAppDispatch();

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <IAIButton onClick={handleReloadSchema}>Reload Schema</IAIButton>
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);
