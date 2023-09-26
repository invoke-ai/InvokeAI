import { Flex, Skeleton, Text } from '@chakra-ui/react';
import { memo } from 'react';
import { COLUMN_WIDTHS } from './constants';

const QueueItemSkeleton = () => {
  return (
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
        <Skeleton width="20px">
          <Text variant="subtext">&nbsp;</Text>
        </Skeleton>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.statusBadge} alignItems="center">
        <Skeleton width="100%">
          <Text variant="subtext">&nbsp;</Text>
        </Skeleton>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.time} alignItems="center">
        <Skeleton width="100%">
          <Text variant="subtext">&nbsp;</Text>
        </Skeleton>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.batchId} alignItems="center">
        <Skeleton width="100%">
          <Text variant="subtext">&nbsp;</Text>
        </Skeleton>
      </Flex>
      <Flex ps={0.5} w={COLUMN_WIDTHS.fieldValues} alignItems="center" flex="1">
        <Skeleton width="100%">
          <Text variant="subtext">&nbsp;</Text>
        </Skeleton>
      </Flex>
    </Flex>
  );
};

export default memo(QueueItemSkeleton);
