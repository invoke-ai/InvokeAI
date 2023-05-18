import { Flex, Heading, Tooltip, Icon } from '@chakra-ui/react';
import { InvocationTemplate } from 'features/nodes/types/types';
import { memo } from 'react';
import { FaInfoCircle } from 'react-icons/fa';

interface IAINodeHeaderProps {
  nodeId: string;
  template: InvocationTemplate;
}

const IAINodeHeader = (props: IAINodeHeaderProps) => {
  const { nodeId, template } = props;
  return (
    <Flex
      borderTopRadius="md"
      justifyContent="space-between"
      background="base.700"
      px={2}
      py={1}
      alignItems="center"
    >
      <Tooltip label={nodeId}>
        <Heading size="xs" fontWeight={600} color="base.100">
          {template.title}
        </Heading>
      </Tooltip>
      <Tooltip
        label={template.description}
        placement="top"
        hasArrow
        shouldWrapChildren
      >
        <Icon color="base.300" as={FaInfoCircle} h="min-content" />
      </Tooltip>
    </Flex>
  );
};

export default memo(IAINodeHeader);
