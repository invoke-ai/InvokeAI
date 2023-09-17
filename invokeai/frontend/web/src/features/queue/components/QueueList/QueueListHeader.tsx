import { Flex, Text } from '@chakra-ui/react';
import { memo } from 'react';
import { COLUMN_WIDTHS } from './constants';

const QueueListHeader = () => {
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
  );
};

export default memo(QueueListHeader);
