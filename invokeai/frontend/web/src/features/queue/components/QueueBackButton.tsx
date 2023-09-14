import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import EnqueueButtonTooltip from './QueueButtonTooltip';

const QueueBackButton = () => {
  const tabName = useAppSelector(activeTabNameSelector);
  const { t } = useTranslation();
  const { isReady } = useIsReadyToEnqueue();
  const dispatch = useAppDispatch();
  const isQueueMutationInProgress = useIsQueueMutationInProgress();

  const handleEnqueue = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName, prepend: false }));
  }, [dispatch, tabName]);

  useHotkeys(
    ['ctrl+enter', 'meta+enter'],
    handleEnqueue,
    {
      enabled: () => !isQueueMutationInProgress,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isQueueMutationInProgress, tabName]
  );
  return (
    <IAIButton
      isDisabled={!isReady}
      colorScheme="accent"
      onClick={handleEnqueue}
      tooltip={<EnqueueButtonTooltip />}
      isLoading={isQueueMutationInProgress}
      flexGrow={3}
      minW={44}
    >
      {t('queue.queueBack')}
    </IAIButton>
  );
};

export default memo(QueueBackButton);
