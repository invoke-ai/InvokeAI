import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addNodePopoverOpened } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

const AddNodeButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleOpenAddNodePopover = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  return (
    <IconButton
      tooltip={t('nodes.addNodeToolTip')}
      aria-label={t('nodes.addNode')}
      icon={<PiPlusBold />}
      onClick={handleOpenAddNodePopover}
      pointerEvents="auto"
    />
  );
};

export default memo(AddNodeButton);
