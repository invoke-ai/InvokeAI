import type { ChakraProps, CollapseProps } from '@invoke-ai/ui-library';
import { Badge, ButtonGroup, Collapse, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import QueueStatusBadge from 'features/queue/components/common/QueueStatusBadge';
import { useDestinationText } from 'features/queue/components/QueueList/useDestinationText';
import { useOriginText } from 'features/queue/components/QueueList/useOriginText';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { useRetryQueueItem } from 'features/queue/hooks/useRetryQueueItem';
import { getSecondsFromTimestamps } from 'features/queue/util/getSecondsFromTimestamps';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectShouldShowCredits } from 'features/system/store/configSlice';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiXBold } from 'react-icons/pi';
import { useSelector } from 'react-redux';
import type { S } from 'services/api/types';

import { COLUMN_WIDTHS } from './constants';
import QueueItemDetail from './QueueItemDetail';
import type { ListContext } from './types';

const selectedStyles = { bg: 'base.700' };

type InnerItemProps = {
  index: number;
  item: S['SessionQueueItem'];
  context: ListContext;
};

const sx: ChakraProps['sx'] = {
  _hover: selectedStyles,
  "&[aria-selected='true']": selectedStyles,
};

const QueueItemComponent = ({ index, item, context }: InnerItemProps) => {
  const { t } = useTranslation();
  const isRetryEnabled = useFeatureStatus('retryQueueItem');
  const handleToggle = useCallback(() => {
    context.toggleQueueItem(item.item_id);
  }, [context, item.item_id]);
  const cancelQueueItem = useCancelQueueItem();
  const onClickCancelQueueItem = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      cancelQueueItem.trigger(item.item_id);
    },
    [cancelQueueItem, item.item_id]
  );
  const retryQueueItem = useRetryQueueItem();
  const onClickRetryQueueItem = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      retryQueueItem.trigger(item.item_id);
    },
    [item.item_id, retryQueueItem]
  );
  const isOpen = useMemo(() => context.openQueueItems.includes(item.item_id), [context.openQueueItems, item.item_id]);

  const executionTime = useMemo(() => {
    if (!item.completed_at || !item.started_at) {
      return;
    }
    const seconds = getSecondsFromTimestamps(item.started_at, item.completed_at);
    return `${seconds}s`;
  }, [item]);

  const shouldShowCredits = useSelector(selectShouldShowCredits);

  const isCanceled = useMemo(() => ['canceled', 'completed', 'failed'].includes(item.status), [item.status]);
  const isFailed = useMemo(() => ['canceled', 'failed'].includes(item.status), [item.status]);
  const isValidationRun = useMemo(() => item.is_api_validation_run === true, [item.is_api_validation_run]);
  const originText = useOriginText(item.origin);
  const destinationText = useDestinationText(item.destination);

  return (
    <Flex
      flexDir="column"
      aria-selected={isOpen}
      fontSize="sm"
      borderRadius="base"
      justifyContent="center"
      sx={sx}
      data-testid="queue-item"
    >
      <Flex minH={9} alignItems="center" gap={4} p={1.5} cursor="pointer" onClick={handleToggle}>
        <Flex w={COLUMN_WIDTHS.number} justifyContent="flex-end" alignItems="center" flexShrink={0}>
          <Text variant="subtext">{index + 1}</Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.statusBadge} alignItems="center" flexShrink={0}>
          <QueueStatusBadge status={item.status} />
        </Flex>
        <Flex w={COLUMN_WIDTHS.origin} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center">
            {originText}
          </Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.destination} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center">
            {destinationText}
          </Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.time} alignItems="center" flexShrink={0}>
          {executionTime || '-'}
        </Flex>
        {shouldShowCredits && (
          <Flex w={COLUMN_WIDTHS.credits} alignItems="center" flexShrink={0}>
            {item.credits || '-'}
          </Flex>
        )}
        <Flex w={COLUMN_WIDTHS.batchId} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center">
            {item.batch_id}
          </Text>
        </Flex>
        <Flex alignItems="center" overflow="hidden" flexGrow={1}>
          {item.field_values && (
            <Flex gap={2} w="full" whiteSpace="nowrap" textOverflow="ellipsis" overflow="hidden">
              {item.field_values
                .filter((v) => v.node_path !== 'metadata_accumulator')
                .map(({ node_path, field_name, value }) => (
                  <Text as="span" key={`${item.item_id}.${node_path}.${field_name}.${value}`}>
                    <Text as="span" fontWeight="semibold">
                      {node_path}.{field_name}
                    </Text>
                    : {JSON.stringify(value)}
                  </Text>
                ))}
            </Flex>
          )}
        </Flex>
        <Flex alignItems="center" w={COLUMN_WIDTHS.validationRun} flexShrink={0}>
          {isValidationRun && <Badge>{t('workflows.builder.publishingValidationRun')}</Badge>}
        </Flex>
        <Flex alignItems="center" w={COLUMN_WIDTHS.actions} pe={3}>
          <ButtonGroup size="xs" variant="ghost">
            {(!isFailed || !isRetryEnabled || isValidationRun) && (
              <IconButton
                onClick={onClickCancelQueueItem}
                isDisabled={isCanceled}
                isLoading={cancelQueueItem.isLoading}
                aria-label={t('queue.cancelItem')}
                icon={<PiXBold />}
              />
            )}
            {isFailed && isRetryEnabled && !isValidationRun && (
              <IconButton
                onClick={onClickRetryQueueItem}
                isLoading={retryQueueItem.isLoading}
                aria-label={t('queue.retryItem')}
                icon={<PiArrowCounterClockwiseBold />}
              />
            )}
          </ButtonGroup>
        </Flex>
      </Flex>

      <Collapse in={isOpen} transition={transition} unmountOnExit={true}>
        <QueueItemDetail queueItem={item} />
      </Collapse>
    </Flex>
  );
};

const transition: CollapseProps['transition'] = {
  enter: { duration: 0.1 },
  exit: { duration: 0.1 },
};

export default memo(QueueItemComponent);
