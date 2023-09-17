import { ButtonGroup, Flex, Heading, Spinner, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import {
  useCancelByBatchIdsMutation,
  useCancelQueueItemMutation,
  useGetBatchStatusQuery,
  useGetQueueItemQuery,
} from 'services/api/endpoints/queue';
import { SessionQueueItemDTO } from 'services/api/types';

type Props = {
  queueItemDTO: SessionQueueItemDTO;
};

const QueueItemComponent = ({ queueItemDTO }: Props) => {
  const { session_id, batch_id, item_id } = queueItemDTO;
  const { t } = useTranslation();
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const [cancelQueueItem, { isLoading: isLoadingCancelQueueItem }] =
    useCancelQueueItemMutation();
  const [cancelByBatchIds, { isLoading: isLoadingCancelByBatchIds }] =
    useCancelByBatchIdsMutation();
  const { isCanceled } = useGetBatchStatusQuery(
    { batch_id: queueItemDTO.batch_id },
    {
      selectFromResult: ({ data }) => {
        if (!data) {
          return { isCanceled: true };
        }

        return {
          isCanceled: data?.in_progress === 0 && data?.pending === 0,
        };
      },
    }
  );
  const handleCancelQueueItem = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      cancelQueueItem(item_id);
    },
    [cancelQueueItem, item_id]
  );
  const handleCancelBatch = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      cancelByBatchIds({ batch_ids: [batch_id] });
    },
    [cancelByBatchIds, batch_id]
  );
  const { data: queueItem } = useGetQueueItemQuery(item_id);
  const executionTime = useMemo(() => {
    if (!queueItem?.completed_at || !queueItem?.started_at) {
      return 'n/a';
    }
    return String(
      (
        (Date.parse(queueItem.completed_at) -
          Date.parse(queueItem.started_at)) /
        1000
      ).toFixed(2)
    );
  }, [queueItem?.completed_at, queueItem?.started_at]);

  return (
    <Flex
      layerStyle="third"
      flexDir="column"
      p={2}
      pt={0}
      borderRadius="base"
      gap={2}
    >
      <Flex
        layerStyle="second"
        p={2}
        gap={2}
        justifyContent="space-between"
        alignItems="center"
        borderRadius="base"
      >
        <QueueItemData label="Item ID" data={item_id} />
        <QueueItemData label="Batch ID" data={batch_id} />
        <QueueItemData label="Session ID" data={session_id} />
        <QueueItemData label="Execution Time" data={executionTime} />
        <ButtonGroup size="xs" orientation="vertical">
          <IAIButton
            onClick={handleCancelQueueItem}
            isLoading={isLoadingCancelQueueItem}
            isDisabled={
              queueItem
                ? ['canceled', 'completed', 'failed'].includes(queueItem.status)
                : true
            }
            aria-label={t('queue.cancelItem')}
            icon={<FaTimes />}
            colorScheme="error"
          >
            {t('queue.cancelItem')}
          </IAIButton>
          <IAIButton
            onClick={handleCancelBatch}
            isLoading={isLoadingCancelByBatchIds || isQueueMutationInProgress}
            isDisabled={isCanceled}
            aria-label={t('queue.cancelBatch')}
            icon={<FaTimes />}
            colorScheme="error"
          >
            {t('queue.cancelBatch')}
          </IAIButton>
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
          <Heading size="sm" color="error.500" _dark={{ color: 'error.400' }}>
            Error
          </Heading>
          <pre>{queueItem.error}</pre>
        </Flex>
      )}
      <Flex
        layerStyle="second"
        h={512}
        w="full"
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
      >
        {queueItem ? (
          <ScrollableContent>
            <DataViewer label="Queue Item" data={queueItem} />
          </ScrollableContent>
        ) : (
          <Spinner opacity={0.5} />
        )}
      </Flex>
    </Flex>
  );
};

export default memo(QueueItemComponent);

type QueueItemDataProps = { label: string; data: string };

const QueueItemData = ({ label, data }: QueueItemDataProps) => {
  return (
    <Flex flexDir="column" p={1} gap={1}>
      <Heading size="sm">{label}</Heading>
      <Text>{data}</Text>
    </Flex>
  );
};
