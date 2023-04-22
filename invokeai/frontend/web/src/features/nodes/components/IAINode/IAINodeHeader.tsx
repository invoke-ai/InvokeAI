import { Flex, Heading, Tooltip, Icon } from '@chakra-ui/react';
import { InvocationTemplate } from 'features/nodes/types/types';
import { MutableRefObject } from 'react';
import { FaInfoCircle } from 'react-icons/fa';

interface IAINodeHeaderProps {
  nodeId: string;
  template: MutableRefObject<InvocationTemplate | undefined>;
}

export default function IAINodeHeader(props: IAINodeHeaderProps) {
  const { nodeId, template } = props;
  return (
    <Flex
      borderRadius="sm"
      justifyContent="space-between"
      background="base.700"
      px={2}
      py={1}
      alignItems="center"
    >
      <Tooltip label={nodeId}>
        <Heading size="sm" fontWeight={600} color="base.100">
          {template.current?.title}
        </Heading>
      </Tooltip>
      <Tooltip
        label={template.current?.description}
        placement="top"
        hasArrow
        shouldWrapChildren
      >
        <Icon color="base.300" as={FaInfoCircle} />
      </Tooltip>
    </Flex>
  );
}
