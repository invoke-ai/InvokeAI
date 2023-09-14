import {
  Box,
  Flex,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  forwardRef,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  listCursorChanged,
  listPriorityChanged,
} from 'features/queue/store/queueSlice';
import {
  UseOverlayScrollbarsParams,
  useOverlayScrollbars,
} from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FixedHeaderContent,
  ItemContent,
  TableComponents,
  TableVirtuoso,
} from 'react-virtuoso';
import {
  queueItemsAdapter,
  useListQueueItemsQuery,
} from 'services/api/endpoints/queue';
import { SessionQueueItemDTO } from 'services/api/types';
import QueueStatusBadge from './common/QueueStatusBadge';

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

const QueueTab = () => {
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

  const { data: listQueueItemsData } = useListQueueItemsQuery(
    {
      cursor: listCursor,
      priority: listPriority,
    },
    { refetchOnMountOrArgChange: true }
  );

  console.log(listCursor, listPriority);

  const queueItems = useMemo(() => {
    if (!listQueueItemsData) {
      return [];
    }
    return queueItemsAdapter.getSelectors().selectAll(listQueueItemsData);
  }, [listQueueItemsData]);

  const handleLoadMore = useCallback(() => {
    dispatch(listCursorChanged(queueItems[queueItems.length - 1]?.id));
    dispatch(listPriorityChanged(queueItems[queueItems.length - 1]?.priority));
  }, [dispatch, queueItems]);

  return (
    <Box ref={rootRef} w="full" h="full">
      {listQueueItemsData && (
        <TableVirtuoso
          data={queueItems}
          endReached={handleLoadMore}
          scrollerRef={setScroller as TableVirtuosoScrollerRef}
          components={TableComponents}
          fixedHeaderContent={FixedHeaderContent}
          itemContent={ItemContent}
        />
      )}
    </Box>
  );
};

export default memo(QueueTab);

const TableComponents: TableComponents<SessionQueueItemDTO> = {
  Table: forwardRef((props, ref) => (
    <Table {...props} ref={ref} size="sm" layout="fixed" />
  )),
  TableHead: forwardRef((props, ref) => <Thead {...props} ref={ref} h={8} />),
  TableRow: forwardRef((props, ref) => (
    <Tr
      {...props}
      ref={ref}
      _hover={{ bg: 'base.150', _dark: { bg: 'base.750' } }}
    />
  )),
  TableBody: forwardRef((props, ref) => <Tbody {...props} ref={ref} />),
};

const FixedHeaderContent: FixedHeaderContent = () => (
  <Tr layerStyle="second">
    <Th border="none" w="4rem" borderLeftRadius="base" textAlign="right">
      <Text variant="subtext">#</Text>
    </Th>
    <Th border="none" w="7.5rem">
      <Text variant="subtext">Status</Text>
    </Th>
    <Th border="none" w="7rem">
      <Text variant="subtext">Batch</Text>
    </Th>
    <Th border="none" borderRightRadius="base">
      <Text variant="subtext">Batch Values</Text>
    </Th>
  </Tr>
);

const ItemContent: ItemContent<SessionQueueItemDTO, unknown> = (
  index,
  { id, status, batch_id, field_values }
) => (
  <>
    <Td
      borderColor="base.150"
      _dark={{ borderColor: 'base.750' }}
      textAlign="right"
    >
      <Text>{index + 1}</Text>
    </Td>
    <Td borderColor="base.150" _dark={{ borderColor: 'base.750' }}>
      <QueueStatusBadge status={status} />
    </Td>
    <Td borderColor="base.150" _dark={{ borderColor: 'base.750' }}>
      <Text overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
        {batch_id}
      </Text>
    </Td>
    <Td borderColor="base.150" _dark={{ borderColor: 'base.750' }}>
      {field_values && (
        <Flex gap={2}>
          {field_values
            .filter((v) => v.node_path !== 'metadata_accumulator')
            .map(({ node_path, field_name, value }) => (
              <Text
                key={`${id}.${node_path}.${field_name}.${value}`}
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
    </Td>
  </>
);
