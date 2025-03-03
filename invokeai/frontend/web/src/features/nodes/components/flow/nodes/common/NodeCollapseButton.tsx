import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Icon, IconButton } from '@invoke-ai/ui-library';
import { useUpdateNodeInternals } from '@xyflow/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { nodeIsOpenChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_FIT_ON_DOUBLE_CLICK_CLASS } from 'features/nodes/types/constants';
import { memo, useCallback } from 'react';
import { PiCaretUpBold } from 'react-icons/pi';

interface Props {
  nodeId: string;
  isOpen: boolean;
}

const iconSx: SystemStyleObject = {
  transitionProperty: 'transform',
  transitionDuration: 'normal',
  transform: 'rotate(180deg)',
  '&[data-is-open="true"]': {
    transform: 'rotate(0deg)',
  },
};

const NodeCollapseButton = ({ nodeId, isOpen }: Props) => {
  const dispatch = useAppDispatch();
  const updateNodeInternals = useUpdateNodeInternals();

  const handleClick = useCallback(() => {
    dispatch(nodeIsOpenChanged({ nodeId, isOpen: !isOpen }));
    updateNodeInternals(nodeId);
  }, [dispatch, isOpen, nodeId, updateNodeInternals]);

  return (
    <IconButton
      className={`${NO_DRAG_CLASS} ${NO_FIT_ON_DOUBLE_CLICK_CLASS}`}
      onClick={handleClick}
      aria-label="Minimize"
      minW={8}
      w={8}
      h={8}
      variant="link"
      icon={<Icon as={PiCaretUpBold} sx={iconSx} data-is-open={isOpen} />}
    />
  );
};

export default memo(NodeCollapseButton);
