import { Button, ButtonGroup, Flex, Heading, Spinner, Text } from '@invoke-ai/ui-library';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useCancelBatch } from 'features/queue/hooks/useCancelBatch';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { getSecondsFromTimestamps } from 'features/queue/util/getSecondsFromTimestamps';
import { get } from 'lodash-es';
import type { ReactNode } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useGetQueueItemQuery } from 'services/api/endpoints/queue';
import type { SessionQueueItemDTO } from 'services/api/types';

type Props = {
  queueItemDTO: SessionQueueItemDTO;
};

const QueueItemComponent = ({ queueItemDTO }: Props) => {
  const { session_id, batch_id, item_id } = queueItemDTO;
  const { t } = useTranslation();
  const { cancelBatch, isLoading: isLoadingCancelBatch, isCanceled } = useCancelBatch(batch_id);

  const { cancelQueueItem, isLoading: isLoadingCancelQueueItem } = useCancelQueueItem(item_id);

  const { data: queueItem } = useGetQueueItemQuery(item_id);

  const statusAndTiming = useMemo(() => {
    if (!queueItem) {
      return t('common.loading');
    }
    if (!queueItem.completed_at || !queueItem.started_at) {
      return t(`queue.${queueItem.status}`);
    }
    const seconds = getSecondsFromTimestamps(queueItem.started_at, queueItem.completed_at);
    if (queueItem.status === 'completed') {
      return `${t('queue.completedIn')} ${seconds}${seconds === 1 ? '' : 's'}`;
    }
    return `${seconds}s`;
  }, [queueItem, t]);

  return (
    <Flex layerStyle="third" flexDir="column" p={2} pt={0} borderRadius="base" gap={2}>
      <Flex
        layerStyle="second"
        p={2}
        gap={2}
        justifyContent="space-between"
        alignItems="center"
        borderRadius="base"
        h={20}
      >
        <QueueItemData label={t('queue.status')} data={statusAndTiming} />
        <QueueItemData label={t('queue.item')} data={item_id} />
        <QueueItemData label={t('queue.batch')} data={batch_id} />
        <QueueItemData label={t('queue.session')} data={session_id} />
        <ButtonGroup size="xs" orientation="vertical">
          <Button
            onClick={cancelQueueItem}
            isLoading={isLoadingCancelQueueItem}
            isDisabled={queueItem ? ['canceled', 'completed', 'failed'].includes(queueItem.status) : true}
            aria-label={t('queue.cancelItem')}
            leftIcon={<PiXBold />}
            colorScheme="error"
          >
            {t('queue.cancelItem')}
          </Button>
          <Button
            onClick={cancelBatch}
            isLoading={isLoadingCancelBatch}
            isDisabled={isCanceled}
            aria-label={t('queue.cancelBatch')}
            leftIcon={<PiXBold />}
            colorScheme="error"
          >
            {t('queue.cancelBatch')}
          </Button>
        </ButtonGroup>
      </Flex>
      {queueItem?.error && (
        <Flex
          layerStyle="second"
          p={3}
          gap={1}
          justifyContent="space-between"
          alignItems="flex-start"
          borderRadius="base"
          flexDir="column"
        >
          <Heading size="sm" color="error.400">
            {t('common.error')}
          </Heading>
          <pre>{queueItem.error}</pre>
        </Flex>
      )}
      <Flex layerStyle="second" h={512} w="full" borderRadius="base" alignItems="center" justifyContent="center">
        {queueItem ? (
          <DataViewer
            label="Queue Item"
            data={queueItem}
            extraCopyActions={[{ label: 'Graph', getData: (data) => get(data, 'session.graph') }]}
          />
        ) : (
          <Spinner opacity={0.5} />
        )}
      </Flex>
    </Flex>
  );
};

export default memo(QueueItemComponent);

type QueueItemDataProps = { label: string; data: ReactNode };

const QueueItemData = ({ label, data }: QueueItemDataProps) => {
  return (
    <Flex flexDir="column" justifyContent="flex-start" p={1} gap={1} overflow="hidden" h="full" w="full">
      <Heading size="md" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
        {label}
      </Heading>
      <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
        {data}
      </Text>
    </Flex>
  );
};
