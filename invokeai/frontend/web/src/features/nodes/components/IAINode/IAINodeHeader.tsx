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
      sx={{
        borderTopRadius: 'md',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 2,
        py: 1,
        bg: 'base.300',
        _dark: { bg: 'base.700' },
      }}
    >
      <Tooltip label={nodeId}>
        <Heading
          size="xs"
          sx={{
            fontWeight: 600,
            color: 'base.900',
            _dark: { color: 'base.100' },
          }}
        >
          {template.title}
        </Heading>
      </Tooltip>
      <Tooltip
        label={template.description}
        placement="top"
        hasArrow
        shouldWrapChildren
      >
        <Icon
          sx={{
            h: 'min-content',
            color: 'base.700',
            _dark: {
              color: 'base.300',
            },
          }}
          as={FaInfoCircle}
        />
      </Tooltip>
    </Flex>
  );
};

export default memo(IAINodeHeader);
