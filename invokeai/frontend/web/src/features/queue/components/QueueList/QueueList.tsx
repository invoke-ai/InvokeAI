import {
  Box,
  ChakraProps,
  Collapse,
  Flex,
  Text,
  forwardRef,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import {
  listCursorChanged,
  listPriorityChanged,
} from 'features/queue/store/queueSlice';
import {
  UseOverlayScrollbarsParams,
  useOverlayScrollbars,
} from 'overlayscrollbars-react';
import {
  MouseEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import { Components, ItemContent, Virtuoso } from 'react-virtuoso';
import {
  queueItemsAdapter,
  useCancelQueueItemMutation,
  useGetQueueItemQuery,
  useListQueueItemsQuery,
} from 'services/api/endpoints/queue';
import { SessionQueueItemDTO } from 'services/api/types';
import QueueStatusBadge from '../common/QueueStatusBadge';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TableVirtuosoScrollerRef = (ref: HTMLElement | Window | null) => any;

const overlayScrollbarsConfig: UseOverlayScrollbarsParams = {
  defer: true,
  options: {
    scrollbars: {
      visibility: 'auto',
      autoHide: 'scroll',
      autoHideDelay: 1300,
      theme: 'os-theme-dark',
    },
    overflow: { x: 'hidden' },
  },
};

const selector = createSelector(
  stateSelector,
  ({ queue }) => {
    const { listCursor, listPriority } = queue;
    return { listCursor, listPriority };
  },
  defaultSelectorOptions
);

const COLUMN_WIDTHS = {
  number: '3rem',
  statusBadge: '5.7rem',
  batchId: '5rem',
  fieldValues: 'auto',
  actions: 'auto',
};

const computeItemKey = (index: number, item: SessionQueueItemDTO): string =>
  item.item_id;

type ListContext = {
  openQueueItems: string[];
  toggleQueueItem: (item_id: string) => void;
};

const ListComponent: Components<SessionQueueItemDTO, ListContext>['List'] =
  memo(
    forwardRef((props, ref) => {
      return (
        <Flex {...props} ref={ref} flexDirection="column">
          {props.children}
        </Flex>
      );
    })
  );

ListComponent.displayName = 'ListComponent';

const components: Components<SessionQueueItemDTO, ListContext> = {
  List: ListComponent,
};

const QueueList = () => {
  const { listCursor, listPriority } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const rootRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(
    overlayScrollbarsConfig
  );

  useEffect(() => {
    const { current: root } = rootRef;
    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }
    return () => osInstance()?.destroy();
  }, [scroller, initialize, osInstance]);

  const { data: listQueueItemsData } = useListQueueItemsQuery({
    cursor: listCursor,
    priority: listPriority,
  });

  const queueItems = useMemo(() => {
    if (!listQueueItemsData) {
      return [];
    }
    return queueItemsAdapter.getSelectors().selectAll(listQueueItemsData);
  }, [listQueueItemsData]);

  const handleLoadMore = useCallback(() => {
    if (!listQueueItemsData?.has_more) {
      return;
    }
    const lastItem = queueItems[queueItems.length - 1];
    if (!lastItem) {
      return;
    }
    dispatch(listCursorChanged(lastItem.order_id));
    dispatch(listPriorityChanged(lastItem.priority));
  }, [dispatch, listQueueItemsData?.has_more, queueItems]);

  const [openQueueItems, setOpenQueueItems] = useState<string[]>([]);

  const toggleQueueItem = useCallback((item_id: string) => {
    setOpenQueueItems((prev) => {
      if (prev.includes(item_id)) {
        return prev.filter((id) => id !== item_id);
      }
      return [...prev, item_id];
    });
  }, []);

  const context = useMemo<ListContext>(
    () => ({ openQueueItems, toggleQueueItem }),
    [openQueueItems, toggleQueueItem]
  );

  return (
    <Box w="full" h="full">
      <Flex
        alignItems="center"
        gap={4}
        p={1}
        pb={2}
        textTransform="uppercase"
        fontWeight={700}
        fontSize="xs"
        letterSpacing={1}
      >
        <Flex
          w={COLUMN_WIDTHS.number}
          justifyContent="flex-end"
          alignItems="center"
        >
          <Text variant="subtext">#</Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.statusBadge} alignItems="center">
          <Text variant="subtext">status</Text>
        </Flex>
        <Flex w={COLUMN_WIDTHS.batchId} alignItems="center">
          <Text variant="subtext">batch</Text>
        </Flex>
        <Flex alignItems="center" w={COLUMN_WIDTHS.fieldValues}>
          <Text variant="subtext">batch field values</Text>
        </Flex>
      </Flex>
      <Box ref={rootRef} w="full" h="full">
        {listQueueItemsData && (
          <Virtuoso<SessionQueueItemDTO, ListContext>
            data={queueItems}
            endReached={handleLoadMore}
            scrollerRef={setScroller as TableVirtuosoScrollerRef}
            itemContent={itemContent}
            computeItemKey={computeItemKey}
            components={components}
            context={context}
            style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}
          />
        )}
      </Box>
    </Box>
  );
};

export default memo(QueueList);

const selectedStyles = { bg: 'base.300', _dark: { bg: 'base.750' } };

const itemContent: ItemContent<SessionQueueItemDTO, ListContext> = (
  index,
  item,
  context
) => <InnerItem index={index} item={item} context={context} />;

type InnerItemProps = {
  index: number;
  item: SessionQueueItemDTO;
  context: ListContext;
};

const sx: ChakraProps['sx'] = {
  _hover: selectedStyles,
  "&[aria-selected='true']": selectedStyles,
};

const InnerItem = memo(({ index, item, context }: InnerItemProps) => {
  const { t } = useTranslation();
  const handleToggle = useCallback(() => {
    context.toggleQueueItem(item.item_id);
  }, [context, item.item_id]);
  const [cancelQueueItem, { isLoading }] = useCancelQueueItemMutation();
  const handleCancel = useCallback(
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
  const { data: queueItem } = useGetQueueItemQuery(
    isOpen ? item.item_id : skipToken
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
        p={1}
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
        <Flex alignItems="center" w={COLUMN_WIDTHS.actions}>
          <IAIIconButton
            tooltip={t('queue.cancelItem')}
            onClick={handleCancel}
            isLoading={isLoading}
            isDisabled={['canceled', 'completed', 'failed'].includes(
              item.status
            )}
            aria-label={t('queue.cancelItem')}
            size="xs"
            variant="ghost"
            icon={<FaTimes />}
          />
        </Flex>
      </Flex>

      <Collapse in={isOpen}>
        <Flex layerStyle="third" p={2} pt={0} borderRadius="base">
          <Flex h={512} w="full" pos="relative">
            {queueItem ? (
              <ScrollableContent>
                <DataViewer label="Queue Item" data={queueItem} />
              </ScrollableContent>
            ) : (
              <IAINoContentFallback label="Loading" icon={null} />
            )}
          </Flex>
        </Flex>
      </Collapse>
    </Flex>
  );
});

InnerItem.displayName = 'InnerItem';
