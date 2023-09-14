import { Flex, Heading, Text, Tooltip } from '@chakra-ui/react';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { memo } from 'react';
import { components } from 'services/api/schema';

const QueueItemCard = ({
  session_queue_item,
  label,
}: {
  session_queue_item?: components['schemas']['SessionQueueItem'];
  label: string;
}) => {
  return (
    <Flex
      layerStyle="second"
      borderRadius="base"
      w="full"
      p={2}
      flexDir="column"
      gap={1}
    >
      <Flex justifyContent="space-between" alignItems="flex-start">
        <Heading size="md">{label}</Heading>
        {session_queue_item && (
          <Tooltip label="Batch ID" placement="top" hasArrow>
            <Text fontSize="xs">{session_queue_item.batch_id}</Text>
          </Tooltip>
        )}
      </Flex>
      {session_queue_item && (
        <ScrollableContent>
          <Text>Batch Values: </Text>
          {session_queue_item.field_values &&
            session_queue_item.field_values
              .filter((v) => v.node_path !== 'metadata_accumulator')
              .map(({ node_path, field_name, value }) => (
                <Text
                  key={`${session_queue_item.id}.${node_path}.${field_name}.${value}`}
                >
                  <Text as="span" fontWeight={600}>
                    {node_path}.{field_name}
                  </Text>
                  : {value}
                </Text>
              ))}
        </ScrollableContent>
      )}
    </Flex>
  );
};

export default memo(QueueItemCard);
