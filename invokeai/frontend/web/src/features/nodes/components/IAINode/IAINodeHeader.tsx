import { Flex, Heading, Icon, Tooltip } from '@chakra-ui/react';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/hooks/useBuildInvocation';
import { memo } from 'react';
import { FaInfoCircle } from 'react-icons/fa';

interface IAINodeHeaderProps {
  nodeId?: string;
  title?: string;
  description?: string;
}

const IAINodeHeader = (props: IAINodeHeaderProps) => {
  const { nodeId, title, description } = props;
  return (
    <Flex
      className={DRAG_HANDLE_CLASSNAME}
      sx={{
        borderTopRadius: 'md',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 2,
        py: 1,
        bg: 'base.100',
        _dark: { bg: 'base.900' },
      }}
    >
      <Tooltip label={nodeId}>
        <Heading
          size="xs"
          sx={{
            fontWeight: 600,
            color: 'base.900',
            _dark: { color: 'base.200' },
          }}
        >
          {title}
        </Heading>
      </Tooltip>
      <Tooltip label={description} placement="top" hasArrow shouldWrapChildren>
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
