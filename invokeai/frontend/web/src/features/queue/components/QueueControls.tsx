import { Button, ButtonGroup, Flex, Spacer } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import CancelCurrentQueueItemButton from 'features/queue/components/CancelCurrentQueueItemButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import PauseProcessorButton from 'features/queue/components/PauseProcessorButton';
import QueueBackButton from 'features/queue/components/QueueBackButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import ResumeProcessorButton from 'features/queue/components/ResumeProcessorButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';

const QueueControls = () => {
  const isPauseEnabled = useFeatureStatus('pauseQueue').isFeatureEnabled;
  const isResumeEnabled = useFeatureStatus('resumeQueue').isFeatureEnabled;
  const isPrependEnabled = useFeatureStatus('prependQueue').isFeatureEnabled;
  return (
    <Flex
      layerStyle="first"
      sx={{
        w: 'full',
        position: 'relative',
        borderRadius: 'base',
        p: 2,
        gap: 2,
        flexDir: 'column',
      }}
    >
      <Flex gap={2} w="full">
        <ButtonGroup isAttached flexGrow={2}>
          <QueueBackButton />
          {isPrependEnabled ? <QueueFrontButton asIconButton /> : <></>}
          <CancelCurrentQueueItemButton asIconButton />
        </ButtonGroup>
        <ButtonGroup isAttached>
          {isResumeEnabled ? <ResumeProcessorButton asIconButton /> : <></>}
          {isPauseEnabled ? <PauseProcessorButton asIconButton /> : <></>}
        </ButtonGroup>
        <ClearQueueButton asIconButton />
      </Flex>
      <Flex h={3} w="full">
        <ProgressBar />
      </Flex>
      <QueueCounts />
    </Flex>
  );
};

export default memo(QueueControls);

const QueueCounts = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { hasItems, pending } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return {
          hasItems: false,
          pending: 0,
        };
      }

      const { pending, in_progress } = data.queue;

      return {
        hasItems: pending + in_progress > 0,
        pending,
      };
    },
  });

  const handleClick = useCallback(() => {
    dispatch(setActiveTab('queue'));
  }, [dispatch]);

  return (
    <Flex justifyContent="space-between" alignItems="center" pe={1}>
      <Spacer />
      <Button
        onClick={handleClick}
        size="sm"
        variant="link"
        fontWeight={400}
        opacity={0.7}
        fontStyle="oblique 10deg"
      >
        {hasItems
          ? t('queue.queuedCount', {
              pending,
            })
          : t('queue.queueEmpty')}
      </Button>
    </Flex>
  );
});

QueueCounts.displayName = 'QueueCounts';
