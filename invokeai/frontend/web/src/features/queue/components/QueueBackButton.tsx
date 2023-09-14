import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import EnqueueButtonTooltip from './QueueButtonTooltip';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';

const QueueBackButton = () => {
  const tabName = useAppSelector(activeTabNameSelector);
  const { t } = useTranslation();
  const { isReady } = useIsReadyToEnqueue();
  const dispatch = useAppDispatch();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const handleEnqueue = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName, prepend: false }));
  }, [dispatch, tabName]);

  useHotkeys(
    ['ctrl+shift+enter', 'meta+shift+enter'],
    handleEnqueue,
    {
      enabled: () => !isLoading,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isLoading, tabName]
  );
  return (
    <IAIButton
      isDisabled={!isReady}
      colorScheme="accent"
      onClick={handleEnqueue}
      tooltip={<EnqueueButtonTooltip />}
      isLoading={isLoading}
      flexGrow={1}
    >
      {t('queue.queueBack')}
    </IAIButton>
  );
};

export default memo(QueueBackButton);
