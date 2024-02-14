import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';

const UpdateNodesButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const nodesNeedUpdate = useGetNodesNeedUpdate();
  const handleClickUpdateNodes = useCallback(() => {
    dispatch(updateAllNodesRequested());
  }, [dispatch]);

  if (!nodesNeedUpdate) {
    return null;
  }

  return (
    <IconButton
      tooltip={t('nodes.updateAllNodes')}
      aria-label={t('nodes.updateAllNodes')}
      icon={<PiWarningBold />}
      onClick={handleClickUpdateNodes}
      pointerEvents="auto"
      colorScheme="warning"
    />
  );
};

export default memo(UpdateNodesButton);
