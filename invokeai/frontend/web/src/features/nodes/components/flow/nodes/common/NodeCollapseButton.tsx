import { Icon, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { nodeIsOpenChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { PiCaretUpBold } from 'react-icons/pi';
import { useUpdateNodeInternals } from 'reactflow';

interface Props {
  nodeId: string;
  isOpen: boolean;
}

const NodeCollapseButton = ({ nodeId, isOpen }: Props) => {
  const dispatch = useAppDispatch();
  const updateNodeInternals = useUpdateNodeInternals();

  const handleClick = useCallback(() => {
    dispatch(nodeIsOpenChanged({ nodeId, isOpen: !isOpen }));
    updateNodeInternals(nodeId);
  }, [dispatch, isOpen, nodeId, updateNodeInternals]);

  return (
    <IconButton
      className="nodrag"
      onClick={handleClick}
      aria-label="Minimize"
      minW={8}
      w={8}
      h={8}
      variant="link"
      icon={
        <Icon
          as={PiCaretUpBold}
          transform={isOpen ? 'rotate(0deg)' : 'rotate(180deg)'}
          transitionProperty="common"
          transitionDuration="normal"
        />
      }
    />
  );
};

export default memo(NodeCollapseButton);
