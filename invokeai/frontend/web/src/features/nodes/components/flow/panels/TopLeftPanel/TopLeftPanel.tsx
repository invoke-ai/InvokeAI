import { Flex } from '@chakra-ui/layout';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();

  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineStart: 2 }}>
      <IAIIconButton
        tooltip="Add Node (Shift+A, Space)"
        aria-label="Add Node"
        icon={<FaPlus />}
        onClick={handleOpenAddNodePopover}
      />
    </Flex>
  );
};

export default memo(TopLeftPanel);
