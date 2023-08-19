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
        aria-label="Add Node"
        tooltip="Add Node"
        onClick={handleOpenAddNodePopover}
        icon={<FaPlus />}
      />
    </Panel>
  );
};

export default memo(TopLeftPanel);
