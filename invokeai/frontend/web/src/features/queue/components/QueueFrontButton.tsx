import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaBoltLightning } from 'react-icons/fa6';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import EnqueueButtonTooltip from './QueueButtonTooltip';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';

const QueueFrontButton = () => {
  const tabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { isReady } = useIsReadyToEnqueue();
  const { t } = useTranslation();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const handleEnqueue = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName, prepend: true }));
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
    <IAIIconButton
      colorScheme="base"
      aria-label={t('queue.queueFront')}
      isDisabled={!isReady}
      onClick={handleEnqueue}
      tooltip={<EnqueueButtonTooltip prepend />}
      isLoading={isLoading}
      icon={<FaBoltLightning />}
    />
  );
};

export default memo(QueueFrontButton);
