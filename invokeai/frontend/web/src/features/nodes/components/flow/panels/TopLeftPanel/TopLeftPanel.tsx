import { Flex } from '@chakra-ui/layout';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';

const TopLeftPanel = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineStart: 2 }}>
      <IAIIconButton
        tooltip={t('nodes.addNodeToolTip')}
        aria-label={t('nodes.addNode')}
        icon={<FaPlus />}
        onClick={handleOpenAddNodePopover}
      />
    </Flex>
  );
};

export default memo(TopLeftPanel);
