import { Tooltip } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { Panel } from 'reactflow';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();

  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <Panel position="top-left">
      <Tooltip label="Add New Node (Shift+A, Space)">
        <IAIButton
          size="sm"
          aria-label="Add Node"
          onClick={handleOpenAddNodePopover}
        >
          Add Node
        </IAIButton>
      </Tooltip>
    </Panel>
  );
};

export default memo(TopLeftPanel);
