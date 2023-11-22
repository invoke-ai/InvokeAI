import { Flex } from '@chakra-ui/layout';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { FaPlus, FaSync } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';
import IAIButton from 'common/components/IAIButton';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { updateAllNodesRequested } from 'features/nodes/store/actions';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const nodesNeedUpdate = useGetNodesNeedUpdate();
  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);
  const handleClickUpdateNodes = useCallback(() => {
    dispatch(updateAllNodesRequested());
  }, [dispatch]);

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineStart: 2 }}>
      <IAIIconButton
        tooltip={t('nodes.addNodeToolTip')}
        aria-label={t('nodes.addNode')}
        icon={<FaPlus />}
        onClick={handleOpenAddNodePopover}
      />
      {nodesNeedUpdate && (
        <IAIButton leftIcon={<FaSync />} onClick={handleClickUpdateNodes}>
          {t('nodes.updateAllNodes')}
        </IAIButton>
      )}
    </Flex>
  );
};

export default memo(TopLeftPanel);
