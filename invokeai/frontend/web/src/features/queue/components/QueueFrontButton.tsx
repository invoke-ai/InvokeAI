import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaBoltLightning } from 'react-icons/fa6';
import { useIsQueueEmpty } from '../hooks/useIsQueueEmpty';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import EnqueueButtonTooltip from './QueueButtonTooltip';

const QueueFrontButton = () => {
  const tabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { isReady } = useIsReadyToEnqueue();
  const { t } = useTranslation();
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const isEmpty = useIsQueueEmpty();
  const handleEnqueue = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName, prepend: true }));
  }, [dispatch, tabName]);

  useHotkeys(
    ['ctrl+shift+enter', 'meta+shift+enter'],
    handleEnqueue,
    {
      enabled: () => !isQueueMutationInProgress,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isQueueMutationInProgress, tabName]
  );
  return (
    <IAIIconButton
      colorScheme="base"
      aria-label={t('queue.queueFront')}
      isDisabled={!isReady || isQueueMutationInProgress || isEmpty}
      onClick={handleEnqueue}
      tooltip={<EnqueueButtonTooltip prepend />}
      icon={<FaBoltLightning />}
    />
  );
};

export default memo(QueueFrontButton);
