import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Panel } from 'reactflow';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();

  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  useHotkeys(['shift+a'], () => {
    handleOpenAddNodePopover();
  });

  return (
    <Panel position="top-left">
      <IAIButton aria-label="Add Node" onClick={handleOpenAddNodePopover}>
        Add Node
      </IAIButton>
    </Panel>
  );
};

export default memo(TopLeftPanel);
