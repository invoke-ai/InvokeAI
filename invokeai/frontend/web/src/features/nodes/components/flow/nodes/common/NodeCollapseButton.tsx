import { ChevronUpIcon } from '@chakra-ui/icons';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { nodeIsOpenChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
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
    <IAIIconButton
      className="nodrag"
      onClick={handleClick}
      aria-label="Minimize"
      sx={{
        minW: 8,
        w: 8,
        h: 8,
        color: 'base.500',
        _dark: {
          color: 'base.500',
        },
        _hover: {
          color: 'base.700',
          _dark: {
            color: 'base.300',
          },
        },
      }}
      variant="link"
      icon={
        <ChevronUpIcon
          sx={{
            transform: isOpen ? 'rotate(0deg)' : 'rotate(180deg)',
            transitionProperty: 'common',
            transitionDuration: 'normal',
          }}
        />
      }
    />
  );
};

export default memo(NodeCollapseButton);
