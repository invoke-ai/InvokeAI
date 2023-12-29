import { useAppDispatch } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPlus } from 'react-icons/fa';

const AddNodeButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <InvIconButton
      tooltip={t('nodes.addNodeToolTip')}
      aria-label={t('nodes.addNode')}
      icon={<FaPlus />}
      onClick={handleOpenAddNodePopover}
      pointerEvents="auto"
    />
  );
};

export default memo(AddNodeButton);
