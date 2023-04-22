import { useAppDispatch } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';
import { nodesGraphBuilt } from 'services/thunks/session';

const TopCenterPanel = () => {
  const dispatch = useAppDispatch();

  const handleInvoke = useCallback(() => {
    dispatch(nodesGraphBuilt());
  }, [dispatch]);

  return (
    <Panel position="top-center">
      <IAIButton colorScheme="accent" onClick={handleInvoke}>
        Will it blend?
      </IAIButton>
    </Panel>
  );
};

export default memo(TopCenterPanel);
