import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { Panel } from 'reactflow';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();

  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <Panel position="top-left">
      <IAIIconButton
        tooltip="Add Node (Shift+A, Space)"
        aria-label="Add Node"
        icon={<FaPlus />}
        onClick={handleOpenAddNodePopover}
      />
    </Panel>
  );
};

export default memo(TopLeftPanel);
