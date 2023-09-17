import {
  ButtonGroup,
  ChakraProps,
  Collapse,
  Flex,
  Text,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import { useCancelQueueItemMutation } from 'services/api/endpoints/queue';
import { SessionQueueItemDTO } from 'services/api/types';
import QueueStatusBadge from '../common/QueueStatusBadge';
import QueueItemDetail from './QueueItemDetail';
import { COLUMN_WIDTHS } from './constants';
import { ListContext } from './types';

const selectedStyles = { bg: 'base.300', _dark: { bg: 'base.750' } };

type InnerItemProps = {
  index: number;
  item: SessionQueueItemDTO;
  context: ListContext;
};

const sx: ChakraProps['sx'] = {
  _hover: selectedStyles,
  "&[aria-selected='true']": selectedStyles,
};

const QueueItemComponent = ({ index, item, context }: InnerItemProps) => {
  const { t } = useTranslation();
  const handleToggle = useCallback(() => {
    context.toggleQueueItem(item.item_id);
  }, [context, item.item_id]);
  const [cancelQueueItem, { isLoading: isLoadingCancelQueueItem }] =
    useCancelQueueItemMutation();
  const handleCancelQueueItem = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      cancelQueueItem(item.item_id);
    },
    [cancelQueueItem, item.item_id]
  );
  const isOpen = useMemo(
    () => context.openQueueItems.includes(item.item_id),
    [context.openQueueItems, item.item_id]
  );

  return (
    <Flex
      flexDir="column"
      borderRadius="base"
      aria-selected={isOpen}
      fontSize="sm"
      justifyContent="center"
      sx={sx}
    >
      <Flex
        alignItems="center"
        gap={4}
        p={1.5}
        cursor="pointer"
        onClick={handleToggle}
      >
        <Flex
          w={COLUMN_WIDTHS.number}
          justifyContent="flex-end"
          alignItems="center"
        >
          <Text variant="subtext">{index + 1}</Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.statusBadge} alignItems="center">
          <QueueStatusBadge status={item.status} />
        </Flex>
        <Flex w={COLUMN_WIDTHS.batchId}>
          <Text
            overflow="hidden"
            textOverflow="ellipsis"
            whiteSpace="nowrap"
            alignItems="center"
          >
            {item.batch_id}
          </Text>
        </Flex>
        <Flex alignItems="center" flexGrow={1}>
          {item.field_values && (
            <Flex gap={2}>
              {item.field_values
                .filter((v) => v.node_path !== 'metadata_accumulator')
                .map(({ node_path, field_name, value }) => (
                  <Text
                    key={`${item.item_id}.${node_path}.${field_name}.${value}`}
                    whiteSpace="nowrap"
                    textOverflow="ellipsis"
                    overflow="hidden"
                  >
                    <Text as="span" fontWeight={600}>
                      {node_path}.{field_name}
                    </Text>
                    : {value}
                  </Text>
                ))}
            </Flex>
          )}
        </Flex>
        <Flex alignItems="center" w={COLUMN_WIDTHS.actions} pe={3}>
          <ButtonGroup size="xs" variant="ghost">
            <IAIIconButton
              tooltip={t('queue.cancelItem')}
              onClick={handleCancelQueueItem}
              isLoading={isLoadingCancelQueueItem}
              isDisabled={['canceled', 'completed', 'failed'].includes(
                item.status
              )}
              aria-label={t('queue.cancelItem')}
              icon={<FaTimes />}
            />
          </ButtonGroup>
        </Flex>
      </Flex>

      <Collapse
        in={isOpen}
        transition={{ enter: { duration: 0.1 }, exit: { duration: 0.1 } }}
        unmountOnExit={true}
      >
        <QueueItemDetail queueItemDTO={item} />
      </Collapse>
    </Flex>
  );
};

export default memo(QueueItemComponent);
