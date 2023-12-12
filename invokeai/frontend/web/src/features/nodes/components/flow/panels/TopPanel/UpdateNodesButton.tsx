import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExclamationTriangle } from 'react-icons/fa';

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
    <IAIButton
      leftIcon={<FaExclamationTriangle />}
      onClick={handleClickUpdateNodes}
      pointerEvents="auto"
    >
      {t('nodes.updateAllNodes')}
    </IAIButton>
  );
};

export default memo(UpdateNodesButton);
