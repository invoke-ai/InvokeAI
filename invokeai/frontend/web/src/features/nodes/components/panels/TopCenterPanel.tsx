import { HStack } from '@chakra-ui/react';
import { userInvoked } from 'app/store/middleware/listenerMiddleware/listeners/userInvoked';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';
import { receivedOpenAPISchema } from 'services/thunks/schema';

const TopCenterPanel = () => {
  const dispatch = useAppDispatch();

  const handleInvoke = useCallback(() => {
    dispatch(userInvoked('nodes'));
  }, [dispatch]);

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <Panel position="top-center">
      <HStack>
        <IAIButton colorScheme="accent" onClick={handleInvoke}>
          Will it blend?
        </IAIButton>
        <IAIButton onClick={handleReloadSchema}>Reload Schema</IAIButton>
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);
