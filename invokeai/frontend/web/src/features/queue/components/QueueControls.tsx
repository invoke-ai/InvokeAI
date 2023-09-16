import { ButtonGroup, Flex, Text } from '@chakra-ui/react';
import ParamRuns from 'features/parameters/components/Parameters/Core/ParamRuns';
import CancelQueueButton from 'features/queue/components/CancelQueueButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import QueueBackButton from 'features/queue/components/QueueBackButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import StartQueueButton from 'features/queue/components/StartQueueButton';
import StopQueueButton from 'features/queue/components/StopQueueButton';
import { usePredictedQueueCounts } from 'features/queue/hooks/usePredictedQueueCounts';
import ProgressBar from 'features/system/components/ProgressBar';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const QueueControls = () => {
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
          <QueueFrontButton />
        </ButtonGroup>
        <ButtonGroup isAttached>
          <StartQueueButton asIconButton />
          <StopQueueButton asIconButton />
          <CancelQueueButton asIconButton />
          <ClearQueueButton asIconButton />
        </ButtonGroup>
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
  const counts = usePredictedQueueCounts();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();

  if (!counts || !queueStatus) {
    return null;
  }

  const { requested, predicted, max_queue_size } = counts;
  const { pending, in_progress } = queueStatus;
  return (
    <Flex justifyContent="space-between" alignItems="center">
      <ParamRuns />
      {/* <Tooltip
        label={
          requested > predicted &&
          t('queue.queueMaxExceeded', {
            requested,
            skip: requested - predicted,
            max_queue_size,
          })
        }
      >
        <Text
          variant="subtext"
          fontSize="sm"
          fontWeight={400}
          fontStyle="oblique 10deg"
          opacity={0.7}
          color={requested > predicted ? 'warning.500' : undefined}
        >
          {t('queue.queueCountPrediction', { predicted })}
        </Text>
      </Tooltip> */}
      <Text
        variant="subtext"
        fontSize="sm"
        fontWeight={400}
        fontStyle="oblique 10deg"
        opacity={0.7}
        pe={1}
      >
        {pending + in_progress > 0
          ? t('queue.queuedCount', {
              pending,
            })
          : t('queue.queueEmpty')}
      </Text>
    </Flex>
  );
});

QueueCounts.displayName = 'QueueCounts';
