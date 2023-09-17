import { ButtonGroup, Flex, Spacer, Text } from '@chakra-ui/react';
import CancelCurrentQueueItemButton from 'features/queue/components/CancelCurrentQueueItemButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import QueueBackButton from 'features/queue/components/QueueBackButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import ResumeProcessorButton from 'features/queue/components/StartQueueButton';
import PauseProcessorButton from 'features/queue/components/StopQueueButton';
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
          <CancelCurrentQueueItemButton asIconButton />
        </ButtonGroup>
        <ButtonGroup isAttached>
          <ResumeProcessorButton asIconButton />
          <PauseProcessorButton asIconButton />
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
  const { hasItems, pending } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return {
          hasItems: false,
          pending: 0,
        };
      }

      const { pending, in_progress } = data;

      return {
        hasItems: pending + in_progress > 0,
        pending,
      };
    },
  });
  const { t } = useTranslation();
  return (
    <Flex justifyContent="space-between" alignItems="center">
      <Spacer />
      <Text
        variant="subtext"
        fontSize="sm"
        fontWeight={400}
        fontStyle="oblique 10deg"
        pe={1}
      >
        {hasItems
          ? t('queue.queuedCount', {
              pending,
            })
          : t('queue.queueEmpty')}
      </Text>
    </Flex>
  );
});

QueueCounts.displayName = 'QueueCounts';
