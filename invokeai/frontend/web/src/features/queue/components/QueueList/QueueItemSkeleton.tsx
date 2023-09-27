import { Flex, Skeleton } from '@chakra-ui/react';
import { memo } from 'react';
import { COLUMN_WIDTHS } from './constants';

const QueueItemSkeleton = () => {
  return (
    <Flex alignItems="center" p={1.5} gap={4} minH={9} h="full" w="full">
      <Flex
        w={COLUMN_WIDTHS.number}
        justifyContent="flex-end"
        alignItems="center"
      >
        <Skeleton w="full" h="full">
          &nbsp;
        </Skeleton>
      </Flex>
      <Flex w={COLUMN_WIDTHS.statusBadge} alignItems="center">
        <Skeleton w="full" h="full">
          &nbsp;
        </Skeleton>
      </Flex>
      <Flex w={COLUMN_WIDTHS.time} alignItems="center">
        <Skeleton w="full" h="full">
          &nbsp;
        </Skeleton>
      </Flex>
      <Flex w={COLUMN_WIDTHS.batchId} alignItems="center">
        <Skeleton w="full" h="full">
          &nbsp;
        </Skeleton>
      </Flex>
      <Flex w={COLUMN_WIDTHS.fieldValues} alignItems="center" flexGrow={1}>
        <Skeleton w="full" h="full">
          &nbsp;
        </Skeleton>
      </Flex>
    </Flex>
  );
};

export default memo(QueueItemSkeleton);
