import type { ChakraProps, CollapseProps, FlexProps } from '@invoke-ai/ui-library';
import { ButtonGroup, Collapse, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import QueueStatusBadge from 'features/queue/components/common/QueueStatusBadge';
import { useDestinationText } from 'features/queue/components/QueueList/useDestinationText';
import { useOriginText } from 'features/queue/components/QueueList/useOriginText';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { useRetryQueueItem } from 'features/queue/hooks/useRetryQueueItem';
import { getSecondsFromTimestamps } from 'features/queue/util/getSecondsFromTimestamps';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiXBold } from 'react-icons/pi';
import type { S } from 'services/api/types';

import { COLUMN_WIDTHS, SYSTEM_USER_ID } from './constants';
import QueueItemDetail from './QueueItemDetail';

const selectedStyles = { bg: 'base.700' };

type InnerItemProps = {
  index: number;
  item: S['SessionQueueItem'];
};

const sx: ChakraProps['sx'] = {
  _hover: selectedStyles,
  "&[aria-selected='true']": selectedStyles,
};

const QueueItemComponent = ({ index, item }: InnerItemProps) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const currentUser = useAppSelector(selectCurrentUser);

  // Check if the current user can view this queue item's details
  const canViewDetails = useMemo(() => {
    // Admins can view all items
    if (currentUser?.is_admin) {
      return true;
    }
    // Users can view their own items
    if (currentUser?.user_id === item.user_id) {
      return true;
    }
    // System items can be viewed by anyone
    if (item.user_id === SYSTEM_USER_ID) {
      return true;
    }
    return false;
  }, [currentUser, item.user_id]);

  const handleToggle = useCallback(() => {
    if (canViewDetails) {
      setIsOpen((s) => !s);
    }
  }, [canViewDetails]);

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

  const executionTime = useMemo(() => {
    if (!item.completed_at || !item.started_at) {
      return;
    }
    const seconds = getSecondsFromTimestamps(item.started_at, item.completed_at);
    return `${seconds}s`;
  }, [item]);

  const isCanceled = useMemo(() => ['canceled', 'completed', 'failed'].includes(item.status), [item.status]);
  const isFailed = useMemo(() => ['canceled', 'failed'].includes(item.status), [item.status]);
  const originText = useOriginText(item.origin);
  const destinationText = useDestinationText(item.destination);

  // Display user name - prefer display_name, fallback to email, then user_id
  const userText = useMemo(() => {
    if (item.user_display_name) {
      return item.user_display_name;
    }
    if (item.user_email) {
      return item.user_email;
    }
    return item.user_id || SYSTEM_USER_ID;
  }, [item.user_display_name, item.user_email, item.user_id]);

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
      <Flex
        minH={9}
        alignItems="center"
        gap={4}
        p={1.5}
        cursor={canViewDetails ? 'pointer' : 'not-allowed'}
        onClick={handleToggle}
        title={!canViewDetails ? t('queue.cannotViewDetails') : undefined}
        opacity={canViewDetails ? 1 : 0.7}
      >
        <Flex w={COLUMN_WIDTHS.number} alignItems="center" flexShrink={0}>
          <Text variant="subtext">{index + 1}</Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.createdAt} alignItems="center" flexShrink={0} flexGrow={0}>
          {new Date(item.created_at).toLocaleString()}
        </Flex>
        <Flex w={COLUMN_WIDTHS.statusBadge} alignItems="center" flexShrink={0}>
          <QueueStatusBadge status={item.status} />
        </Flex>

        <Flex w={COLUMN_WIDTHS.time} alignItems="center" flexShrink={0}>
          {executionTime || '-'}
        </Flex>
        <Flex w={COLUMN_WIDTHS.origin_destination} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center">
            {originText} / {destinationText}
          </Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.batchId} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center">
            {item.batch_id}
          </Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.user} flexShrink={0}>
          <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap" alignItems="center" title={userText}>
            {userText}
          </Text>
        </Flex>
        <Flex overflow="hidden" flexGrow={1}>
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
          {!item.field_values && item.user_id !== SYSTEM_USER_ID && (
            <Text as="span" color="base.500" fontStyle="italic">
              {t('queue.fieldValuesHidden')}
            </Text>
          )}
        </Flex>

        <Flex alignItems="center" w={COLUMN_WIDTHS.actions} pe={3}>
          <ButtonGroup size="xs" variant="ghost">
            {!isFailed && (
              <IconButton
                onClick={onClickCancelQueueItem}
                isDisabled={isCanceled}
                isLoading={cancelQueueItem.isLoading}
                aria-label={t('queue.cancelItem')}
                icon={<PiXBold />}
              />
            )}
            {isFailed && (
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

export const QueueItemPlaceholder = memo((props: FlexProps) => (
  <Flex h={9} w="full" bg="base.800" borderRadius="base" alignItems="center" justifyContent="center" {...props}></Flex>
));

QueueItemPlaceholder.displayName = 'QueueItemPlaceholder';
