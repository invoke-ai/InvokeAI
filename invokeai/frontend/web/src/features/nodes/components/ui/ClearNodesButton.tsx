import { useAppDispatch } from 'app/store/storeHooks';
import { clearNodes } from 'features/nodes/store/nodesSlice';
import { makeToast } from 'app/components/Toaster';
import { addToast } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import IAIIconButton from 'common/components/IAIIconButton';

const ClearNodesButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleClearNodes = () => {
    const confirmed = window.confirm(t('common.clearNodes'));

    if (confirmed) {
      dispatch(clearNodes());

      dispatch(
        addToast(
          makeToast({
            title: t('toast.nodesCleared'),
            status: 'success',
          })
        )
      );
    }
  };

  return (
    <IAIIconButton
      icon={<FaTrash />}
      tooltip={t('nodes.clearNodes')}
      aria-label={t('nodes.clearNodes')}
      onClick={handleClearNodes}
    />
  );
};

export default memo(ClearNodesButton);
